from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Label(str, Enum):
    """Message category - a flat 4-way taxonomy (no fraud-subtype/
    legitimate hierarchy). OTP is rule-detected (see otp_rule.py) and
    never predicted by the ML model; SPAM/REKLAM/BILGILENDIRME are the
    model's 3 output classes.

    - otp: one-time passwords / verification / 2FA codes
    - reklam: promotional campaigns, discounts, marketing, ads
    - spam: unsolicited, suspicious, phishing, or irrelevant junk
    - bilgilendirme: transactional notifications, account updates,
      shipping status, appointment reminders, general information
      (excluding OTPs)
    """

    OTP = "otp"
    REKLAM = "reklam"
    SPAM = "spam"
    BILGILENDIRME = "bilgilendirme"


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
