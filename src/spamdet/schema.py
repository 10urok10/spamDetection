from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Label(str, Enum):
    """Message category.

    Public source datasets only distinguish spam/ham, so they map to
    LEGITIMATE/SPAM. The fine-grained fraud subtypes are only produced by
    our own synthetic data (no public Turkish dataset for them exists).
    """

    LEGITIMATE = "legitimate"
    SPAM = "spam"
    GAMBLING_SCAM = "gambling_scam"
    PHISHING = "phishing"
    FINANCIAL_URGENCY = "financial_urgency"

    @property
    def coarse(self) -> "Label":
        """Collapse fine-grained synthetic categories to spam/legitimate so
        binary-labeled public datasets and multi-class synthetic data can be
        trained on together (e.g. as an auxiliary binary head in Stage 2)."""
        return self if self is Label.LEGITIMATE else Label.SPAM


class Lang(str, Enum):
    TR = "tr"
    EN = "en"


class Record(BaseModel):
    text: str = Field(..., min_length=1)
    label: Label
    source: str
    lang: Lang
    extra: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must not be blank after stripping")
        return stripped
