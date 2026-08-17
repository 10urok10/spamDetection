from dataclasses import dataclass

from ..model.inference import PredictionResult, SpamClassifier
from ..otp_rule import detect_otp
from ..outbreak.detector import OutbreakDetector, OutbreakResult
from ..preprocessing.homoglyphs import detect_mixed_script, strip_confusables
from ..preprocessing.url_tools import extract_urls
from ..preprocessing.zero_width import strip_zero_width
from ..schema import Label
from .config import REVIEW_CONFIDENCE_HIGH, REVIEW_CONFIDENCE_LOW

# OTP is a deterministic rule match - essentially certain, so it's given
# a fixed high confidence rather than a fabricated/absent number.
_OTP_CONFIDENCE = 0.95


def needs_human_review(confidence: float) -> bool:
    return REVIEW_CONFIDENCE_LOW <= confidence <= REVIEW_CONFIDENCE_HIGH


@dataclass(frozen=True)
class PipelineResult:
    message_id: str
    text: str
    cleaned_text: str
    prediction: PredictionResult
    homoglyph_attack_detected: bool
    urls_found: list[str]
    needs_review: bool
    outbreak: OutbreakResult | None


class ClassificationPipeline:
    """Ties Stage 1 preprocessing, the OTP rule, the ML classifier
    (bilgilendirme/reklam/spam), and the outbreak detector into the
    single per-message flow the API exposes.

    Flat 4-category taxonomy (otp/reklam/spam/bilgilendirme, no fraud-
    subtype/legitimate hierarchy) - see docs/model.md for how this
    replaced the earlier 5-label + legitimate-subtype design. OTP is
    checked first via a deterministic rule (otp_rule.detect_otp) and
    never reaches the ML model - it's templated/structured enough that a
    rule is more reliable than a learned class.

    URL unshortening (built and tested in Stage 1) is intentionally NOT
    called synchronously here - resolving a redirect chain over the
    network can take seconds, which would make /classify's latency
    depend on an attacker-controlled server. Raw extracted URLs are
    still reported; unshortening is better suited to an async background
    enrichment step, noted in docs/production_readiness.md rather than
    built now.
    """

    def __init__(self, classifier: SpamClassifier, outbreak_detector: OutbreakDetector | None = None):
        self.classifier = classifier
        self.outbreak_detector = outbreak_detector

    def process(self, message_id: str, text: str) -> PipelineResult:
        cleaned = strip_zero_width(text)
        homoglyph_report = detect_mixed_script(cleaned)
        if homoglyph_report.is_mixed_script:
            cleaned = strip_confusables(cleaned)
        urls = extract_urls(cleaned)

        if detect_otp(cleaned):
            prediction = PredictionResult(
                label=Label.OTP.value,
                confidence=_OTP_CONFIDENCE,
                probabilities={Label.OTP.value: _OTP_CONFIDENCE},
            )
        else:
            prediction = self.classifier.predict(cleaned)

        outbreak_result = None
        if self.outbreak_detector is not None:
            outbreak_result = self.outbreak_detector.ingest(message_id, text)

        return PipelineResult(
            message_id=message_id,
            text=text,
            cleaned_text=cleaned,
            prediction=prediction,
            homoglyph_attack_detected=homoglyph_report.is_mixed_script,
            urls_found=urls,
            needs_review=needs_human_review(prediction.confidence),
            outbreak=outbreak_result,
        )
