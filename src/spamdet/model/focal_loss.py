from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017): down-weights easy,
    already-well-classified examples so the loss stays dominated by hard/
    minority-class examples - used here because gambling_scam/phishing/
    financial_urgency have far fewer examples than legitimate/spam.
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(label_ids: Sequence[int], num_labels: int) -> torch.Tensor:
    """Inverse-frequency class weights: weight_c = N / (num_labels * count_c).
    Classes absent from ``label_ids`` (shouldn't happen on the training
    split, but guard anyway) get a count of 1 to avoid a division by zero.
    """
    counts = [0] * num_labels
    for label_id in label_ids:
        counts[label_id] += 1
    total = len(label_ids)
    weights = [total / (num_labels * max(count, 1)) for count in counts]
    return torch.tensor(weights, dtype=torch.float)


class FocalLossTrainer(Trainer):
    def __init__(self, *args, focal_loss: FocalLoss, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = focal_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs[0]
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss
