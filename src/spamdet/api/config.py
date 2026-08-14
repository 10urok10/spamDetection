import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DEFAULT_PYTORCH_MODEL_DIR = PROJECT_ROOT / "models" / "spamdet-mdeberta"
DEFAULT_ONNX_MODEL_DIR = PROJECT_ROOT / "models" / "spamdet-mdeberta-onnx"
DEFAULT_SUBTYPE_MODEL_PATH = PROJECT_ROOT / "models" / "subtype-ad-info.joblib"

# Confidence band routed to human review, per the project spec - not a
# tunable we invented, so it's a constant rather than an env var.
REVIEW_CONFIDENCE_LOW = 0.4
REVIEW_CONFIDENCE_HIGH = 0.6

# How confident the subtype ad/info classifier must be before it's
# allowed to overturn the top-level model's own "spam" verdict into
# "legitimate" (see api/pipeline.py's _should_override_spam_verdict).
# Calibrated against real examples, not guessed: a first attempt at 0.75
# missed most genuine ads (real P(reklam) scores range ~0.51-0.81, with
# the flagship real "Hepsiburada kupon" example itself at only 0.64),
# while bilgilendirme examples measured comfortably lower (~0.27-0.37).
# 0.5 clears that whole reklam range with margin to spare below it -
# still higher than AdInfoClassifier's own reklam_threshold (0.4, tuned
# for recall within already-"legitimate" text), since overriding a
# "spam" call is a bigger claim than refining a "legitimate" one, but
# not so high that it misses real ads for the sake of extra caution the
# data doesn't actually call for. The override is also lower-stakes than
# it sounds: it only ever reconsiders generic "spam", never the fraud
# labels (phishing/gambling_scam/financial_urgency) - worst case here is
# mislabeling one mild category as another, not missing real fraud.
SPAM_TO_REKLAM_OVERRIDE_THRESHOLD = 0.5


def get_model_dir() -> Path:
    """SPAMDET_MODEL_DIR env var if set, else the ONNX export if present
    (faster inference), else the raw PyTorch checkpoint."""
    env_value = os.environ.get("SPAMDET_MODEL_DIR")
    if env_value:
        return Path(env_value)
    if DEFAULT_ONNX_MODEL_DIR.exists():
        return DEFAULT_ONNX_MODEL_DIR
    return DEFAULT_PYTORCH_MODEL_DIR


def get_redis_url() -> str:
    return os.environ.get("SPAMDET_REDIS_URL", "redis://localhost:6379/0")


def get_confirmed_data_path() -> Path:
    env_value = os.environ.get("SPAMDET_CONFIRMED_DATA_PATH")
    if env_value:
        return Path(env_value)
    return PROJECT_ROOT / "data" / "review" / "confirmed.jsonl"


def get_subtype_model_path() -> Path:
    """Path to the trained reklam/bilgilendirme AdInfoClassifier joblib
    file. If missing, the API runs without subtype detection (OTP rule
    still works standalone) rather than failing to start - see
    api/app.py's lifespan handler.
    """
    env_value = os.environ.get("SPAMDET_SUBTYPE_MODEL_PATH")
    if env_value:
        return Path(env_value)
    return DEFAULT_SUBTYPE_MODEL_PATH
