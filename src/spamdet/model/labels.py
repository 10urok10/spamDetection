from ..schema import Label

# Fixed label order shared by training, ONNX export, and inference so the
# integer class ids are stable across the whole pipeline.
#
# OTP is intentionally excluded: it's detected by a deterministic rule
# (otp_rule.detect_otp) before the ML model ever runs, not predicted by
# it - OTP messages are templated/structured enough that a rule is more
# reliable (and needs no training data) than a learned class. See
# model/dataset.py's build_training_dataframe(), which filters otp-
# labeled seed data out of the ML training set for the same reason.
LABELS: tuple[Label, ...] = (
    Label.BILGILENDIRME,
    Label.REKLAM,
    Label.SPAM,
)

LABEL2ID: dict[str, int] = {label.value: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label.value for i, label in enumerate(LABELS)}
