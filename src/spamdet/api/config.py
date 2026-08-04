import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DEFAULT_PYTORCH_MODEL_DIR = PROJECT_ROOT / "models" / "spamdet-mdeberta"
DEFAULT_ONNX_MODEL_DIR = PROJECT_ROOT / "models" / "spamdet-mdeberta-onnx"

# Confidence band routed to human review, per the project spec - not a
# tunable we invented, so it's a constant rather than an env var.
REVIEW_CONFIDENCE_LOW = 0.4
REVIEW_CONFIDENCE_HIGH = 0.6


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
