import argparse
import json
from pathlib import Path

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from ..synthetic.augment import TemplateParaphraser
from ..synthetic.seeds import load_subtype_training_data
from .ad_info_classifier import BILGILENDIRME, REKLAM, AdInfoClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_SEED_DIR = PROJECT_ROOT / "data" / "synthetic" / "seeds"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "subtype-ad-info.joblib"


def _augment(pairs: list[tuple[str, str]], *, n_per_seed: int, rng_seed: int) -> tuple[list[str], list[str]]:
    paraphraser = TemplateParaphraser(rng_seed=rng_seed)
    texts: list[str] = []
    labels: list[str] = []
    for text, subtype in pairs:
        texts.append(text)
        labels.append(subtype)
        for variant in paraphraser.paraphrase(text, n=n_per_seed):
            texts.append(variant)
            labels.append(subtype)
    return texts, labels


def load_raw_pairs(seed_dir: Path) -> list[tuple[str, str]]:
    """reklam/bilgilendirme (text, label) pairs from the tagged seeds in
    data/synthetic/seeds/legitimate.yaml, before augmentation. otp-tagged
    entries are excluded - that subtype is rule-detected
    (subtype.rules.detect_otp), not part of this classifier.
    """
    return [(t, s) for t, s in load_subtype_training_data(seed_dir) if s in (REKLAM, BILGILENDIRME)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the reklam-vs-bilgilendirme subtype classifier.")
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--reklam-threshold",
        type=float,
        default=0.4,
        help="lower than 0.5 to favor reklam recall over precision (compliance-audit use case)",
    )
    parser.add_argument("--n-per-seed", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--rng-seed", type=int, default=42)
    args = parser.parse_args(argv)

    raw_pairs = load_raw_pairs(args.seed_dir)
    if not raw_pairs:
        print(f"No reklam/bilgilendirme subtype training data found under {args.seed_dir}")
        return 1
    raw_texts = [t for t, _ in raw_pairs]
    raw_labels = [s for _, s in raw_pairs]
    print(f"raw seeds: {len(raw_pairs)} (reklam={raw_labels.count(REKLAM)}, bilgilendirme={raw_labels.count(BILGILENDIRME)})")

    # Split at the SEED level, before augmentation - paraphrase variants of
    # the same seed must never cross the train/test boundary, or the
    # reported metrics are leaked/optimistic (this is exactly the bug
    # documented in docs/model.md for the main fraud classifier's first
    # training round; not repeating it here).
    train_pairs, test_pairs = train_test_split(
        raw_pairs, test_size=args.test_size, random_state=args.rng_seed, stratify=raw_labels
    )

    x_train, y_train = _augment(train_pairs, n_per_seed=args.n_per_seed, rng_seed=args.rng_seed)
    x_test, y_test = _augment(test_pairs, n_per_seed=args.n_per_seed, rng_seed=args.rng_seed)
    print(f"augmented: train={len(x_train)}, test={len(x_test)}")

    clf = AdInfoClassifier(reklam_threshold=args.reklam_threshold)
    clf.fit(x_train, y_train)

    y_pred = [clf.predict(t).subtype for t in x_test]
    report_text = classification_report(y_test, y_pred, digits=3)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    print(f"\ntest set: {len(x_test)} examples, reklam_threshold={args.reklam_threshold}\n")
    print(report_text)

    clf.save(args.model_path)
    metrics_path = args.model_path.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    print(f"\nModel saved to {args.model_path}")
    print(f"Metrics saved to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
