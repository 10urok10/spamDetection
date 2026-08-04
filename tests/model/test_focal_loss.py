import torch
import torch.nn.functional as F

from spamdet.model.focal_loss import FocalLoss, FocalLossTrainer, compute_class_weights


def test_focal_loss_matches_cross_entropy_when_gamma_zero():
    torch.manual_seed(0)
    logits = torch.randn(8, 4)
    targets = torch.randint(0, 4, (8,))
    focal = FocalLoss(gamma=0.0)
    expected = F.cross_entropy(logits, targets)
    torch.testing.assert_close(focal(logits, targets), expected)


def test_focal_loss_downweights_easy_examples_relative_to_hard_ones():
    easy_logits = torch.tensor([[10.0, -10.0]])  # very confident, correct
    hard_logits = torch.tensor([[0.1, -0.1]])  # barely correct, uncertain
    target = torch.tensor([0])
    focal = FocalLoss(gamma=2.0, reduction="none")
    easy_loss = focal(easy_logits, target)
    hard_loss = focal(hard_logits, target)
    assert easy_loss.item() < hard_loss.item()


def test_focal_loss_applies_class_alpha_weight():
    logits = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    targets = torch.tensor([0, 1])
    alpha = torch.tensor([1.0, 5.0])
    focal = FocalLoss(alpha=alpha, gamma=0.0, reduction="none")
    losses = focal(logits, targets)
    unweighted = FocalLoss(gamma=0.0, reduction="none")(logits, targets)
    torch.testing.assert_close(losses[0], unweighted[0])
    torch.testing.assert_close(losses[1], unweighted[1] * 5.0)


def test_compute_class_weights_gives_rare_classes_higher_weight():
    # class 0: 90 examples, class 1: 10 examples
    label_ids = [0] * 90 + [1] * 10
    weights = compute_class_weights(label_ids, num_labels=2)
    assert weights[1] > weights[0]


def test_compute_class_weights_handles_absent_class_without_error():
    label_ids = [0, 0, 0]
    weights = compute_class_weights(label_ids, num_labels=3)
    assert weights.shape[0] == 3
    assert torch.isfinite(weights).all()


class _FakeOutputs(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


class _FakeModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits

    def forward(self, **kwargs):
        return _FakeOutputs(logits=self._logits)


def test_focal_loss_trainer_compute_loss_uses_focal_loss(monkeypatch):
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    labels = torch.tensor([0, 1])
    model = _FakeModel(logits)
    focal = FocalLoss(gamma=2.0)

    trainer = FocalLossTrainer.__new__(FocalLossTrainer)  # skip full Trainer.__init__ (needs TrainingArguments etc.)
    trainer.focal_loss = focal

    loss = trainer.compute_loss(model, {"labels": labels, "input_ids": torch.zeros(2, 3, dtype=torch.long)})
    expected = focal(logits, labels)
    torch.testing.assert_close(loss, expected)


def test_focal_loss_trainer_compute_loss_return_outputs():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])
    model = _FakeModel(logits)
    focal = FocalLoss(gamma=2.0)

    trainer = FocalLossTrainer.__new__(FocalLossTrainer)
    trainer.focal_loss = focal

    loss, outputs = trainer.compute_loss(
        model, {"labels": labels, "input_ids": torch.zeros(2, 3, dtype=torch.long)}, return_outputs=True
    )
    assert torch.equal(outputs["logits"], logits)
