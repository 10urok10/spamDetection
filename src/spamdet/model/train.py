import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
)

from .dataset import DEFAULT_SEED_DIR, build_training_dataframe, split_dataset, write_split
from .focal_loss import FocalLoss, FocalLossTrainer, compute_class_weights
from .labels import ID2LABEL, LABEL2ID, LABELS
from ..build_dataset import DEFAULT_RAW_DIR
from ..preprocessing.mersis_marker import mark_mersis

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "spamdet-mdeberta"
DEFAULT_MODEL_NAME = "microsoft/mdeberta-v3-base"


def _to_hf_dataset(df, tokenizer, max_length: int) -> Dataset:
    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
    ds = ds.map(lambda batch: {"labels": [LABEL2ID[label] for label in batch["label"]]}, batched=True)

    def _tokenize(batch):
        # mark_mersis is applied here (not baked into the stored
        # text/parquet splits) so it's a pure tokenizer-input transform -
        # see preprocessing/mersis_marker.py. inference.py applies the
        # exact same function at serve time, so train/serve stay in sync.
        marked_texts = [mark_mersis(t) for t in batch["text"]]
        return tokenizer(marked_texts, truncation=True, max_length=max_length)

    ds = ds.map(_tokenize, batched=True)
    return ds.remove_columns(["text", "label"])


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    accuracy = float((preds == labels).mean())
    return {"accuracy": accuracy, "precision_macro": precision, "recall_macro": recall, "f1_macro": f1}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a Turkish spam/fraud classifier with focal loss.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--offline", action="store_true", help="skip turkishsms_ds (no network access)")
    parser.add_argument("--n-per-seed", type=int, default=3)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--rng-seed", type=int, default=42)
    args = parser.parse_args(argv)

    df = build_training_dataframe(
        raw_dir=args.raw_dir,
        seed_dir=args.seed_dir,
        include_turkishsms_ds=not args.offline,
        n_per_seed=args.n_per_seed,
        augment_rng_seed=args.rng_seed,
    )
    if df.empty:
        print("No training data available - place raw datasets under data/raw/ or check data/synthetic/seeds/.")
        return 1

    train_df, val_df, test_df = split_dataset(
        df, val_size=args.val_size, test_size=args.test_size, random_state=args.rng_seed
    )
    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        write_split(split_df, PROJECT_ROOT / "data" / "processed" / "train" / f"{name}.parquet")
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = _to_hf_dataset(train_df, tokenizer, args.max_length)
    val_ds = _to_hf_dataset(val_df, tokenizer, args.max_length)
    test_ds = _to_hf_dataset(test_df, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    class_weights = compute_class_weights(train_ds["labels"], num_labels=len(LABELS))
    focal_loss = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)

    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=use_fp16,
        logging_steps=20,
        report_to=[],
        seed=args.rng_seed,
    )

    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
        focal_loss=focal_loss,
    )

    trainer.train()

    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    print("Test metrics:", test_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    with open(args.output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    with open(args.output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2, ensure_ascii=False)

    print(f"Model saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
