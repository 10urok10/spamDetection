import argparse
import shutil
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from .train import DEFAULT_OUTPUT_DIR

DEFAULT_ONNX_DIR = DEFAULT_OUTPUT_DIR.parent / "spamdet-mdeberta-onnx"


def export_to_onnx(model_dir: str | Path, onnx_dir: str | Path) -> Path:
    model_dir = Path(model_dir)
    onnx_dir = Path(onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    ort_model = ORTModelForSequenceClassification.from_pretrained(model_dir, export=True)
    ort_model.save_pretrained(onnx_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.save_pretrained(onnx_dir)

    label_map_path = model_dir / "label_map.json"
    if label_map_path.exists():
        shutil.copy(label_map_path, onnx_dir / "label_map.json")

    return onnx_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a fine-tuned model to ONNX for faster inference.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    args = parser.parse_args(argv)

    if not args.model_dir.exists():
        print(f"{args.model_dir} does not exist - run scripts/train_model.py first.")
        return 1

    onnx_dir = export_to_onnx(args.model_dir, args.onnx_dir)
    print(f"Exported ONNX model to {onnx_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
