from dataclasses import dataclass

from .ad_info_classifier import AdInfoClassifier
from .rules import detect_otp

OTP = "otp"


@dataclass(frozen=True)
class SubtypeResult:
    subtype: str  # "otp" | "bilgilendirme" | "reklam"
    source: str  # "rule_otp" | "model"
    reklam_probability: float | None = None  # only set when source == "model"


class SubtypeDetector:
    """Classifies WHICH KIND of legitimate message this is - only
    meaningful for messages the top-level model (spamdet.model) already
    called 'legitimate'. Built for an SMS-operator compliance use case:
    telling apart OTP / informational / advertisement traffic, so
    advertisement content can be checked against the right (separately
    consented) sending channel.
    """

    def __init__(self, ad_info_classifier: AdInfoClassifier):
        self.ad_info_classifier = ad_info_classifier

    def detect(self, text: str) -> SubtypeResult:
        if detect_otp(text):
            return SubtypeResult(subtype=OTP, source="rule_otp")
        prediction = self.ad_info_classifier.predict(text)
        return SubtypeResult(
            subtype=prediction.subtype, source="model", reklam_probability=prediction.reklam_probability
        )
