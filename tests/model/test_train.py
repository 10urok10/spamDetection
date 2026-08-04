import numpy as np

from spamdet.model.train import _compute_metrics


def test_compute_metrics_perfect_predictions():
    logits = np.array([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]])
    labels = np.array([0, 1, 0])
    metrics = _compute_metrics((logits, labels))
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_compute_metrics_detects_misclassification():
    logits = np.array([[5.0, 0.0], [5.0, 0.0]])  # both predicted class 0
    labels = np.array([0, 1])  # second is actually class 1
    metrics = _compute_metrics((logits, labels))
    assert metrics["accuracy"] == 0.5
    assert metrics["f1_macro"] < 1.0
