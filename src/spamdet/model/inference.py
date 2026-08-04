import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    probabilities: dict[str, float]


class SpamClassifier:
    """Loads a fine-tuned model (PyTorch checkpoint or ONNX export
    directory, whichever ``model_dir`` contains) and exposes single/batch
    prediction. Both directories are produced with the same layout by
    train.py / export_onnx.py, so this class picks the backend
    automatically based on what's present.
    """

    def __init__(self, model_dir: str | Path, *, max_length: int = 128):
        model_dir = Path(model_dir)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        with open(model_dir / "label_map.json", encoding="utf-8") as f:
            label_map = json.load(f)
        self.id2label: dict[int, str] = {int(k): v for k, v in label_map["id2label"].items()}

        if (model_dir / "model.onnx").exists():
            from optimum.onnxruntime import ORTModelForSequenceClassification

            self.model = ORTModelForSequenceClassification.from_pretrained(model_dir)
            self._backend = "onnx"
        else:
            from transformers import AutoModelForSequenceClassification

            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.eval()
            self._backend = "pytorch"

    @property
    def backend(self) -> str:
        return self._backend

    def predict(self, text: str) -> PredictionResult:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True, max_length=self.max_length, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

        results = []
        for row in probs:
            pred_id = int(torch.argmax(row).item())
            probabilities = {self.id2label[i]: float(p) for i, p in enumerate(row)}
            results.append(
                PredictionResult(label=self.id2label[pred_id], confidence=float(row[pred_id]), probabilities=probabilities)
            )
        return results
