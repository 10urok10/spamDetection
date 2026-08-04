from ..schema import Label

# Fixed label order shared by training, ONNX export, and inference so the
# integer class ids are stable across the whole pipeline.
LABELS: tuple[Label, ...] = (
    Label.LEGITIMATE,
    Label.SPAM,
    Label.GAMBLING_SCAM,
    Label.PHISHING,
    Label.FINANCIAL_URGENCY,
)

LABEL2ID: dict[str, int] = {label.value: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label.value for i, label in enumerate(LABELS)}
