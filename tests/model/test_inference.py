import json
from unittest.mock import MagicMock, patch

import torch

from spamdet.model.inference import SpamClassifier


def _make_model_dir(tmp_path, name="model"):
    d = tmp_path / name
    d.mkdir()
    label_map = {"label2id": {"legitimate": 0, "spam": 1}, "id2label": {"0": "legitimate", "1": "spam"}}
    (d / "label_map.json").write_text(json.dumps(label_map), encoding="utf-8")
    return d


def _fake_tokenizer(batch_size: int):
    return MagicMock(
        return_value={
            "input_ids": torch.zeros(batch_size, 3, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
        }
    )


def test_pytorch_backend_selected_when_no_onnx_file(tmp_path):
    model_dir = _make_model_dir(tmp_path)
    fake_model = MagicMock(return_value=MagicMock(logits=torch.tensor([[3.0, 0.0], [0.0, 3.0]])))

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer(2)),
        patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=fake_model),
    ):
        clf = SpamClassifier(model_dir)

    assert clf.backend == "pytorch"
    results = clf.predict_batch(["a", "b"])
    assert results[0].label == "legitimate"
    assert results[1].label == "spam"
    assert 0.0 <= results[0].confidence <= 1.0


def test_onnx_backend_selected_when_model_onnx_present(tmp_path):
    model_dir = _make_model_dir(tmp_path)
    (model_dir / "model.onnx").write_bytes(b"fake-onnx-bytes")
    fake_ort_model = MagicMock(return_value=MagicMock(logits=torch.tensor([[0.0, 5.0]])))

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer(1)),
        patch("optimum.onnxruntime.ORTModelForSequenceClassification.from_pretrained", return_value=fake_ort_model),
    ):
        clf = SpamClassifier(model_dir)

    assert clf.backend == "onnx"
    result = clf.predict("some text")
    assert result.label == "spam"


def test_predict_returns_probabilities_summing_to_one(tmp_path):
    model_dir = _make_model_dir(tmp_path)
    fake_model = MagicMock(return_value=MagicMock(logits=torch.tensor([[1.0, 2.0]])))

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer(1)),
        patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=fake_model),
    ):
        clf = SpamClassifier(model_dir)

    result = clf.predict("x")
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-5
    assert result.label == max(result.probabilities, key=result.probabilities.get)
    assert result.confidence == result.probabilities[result.label]
