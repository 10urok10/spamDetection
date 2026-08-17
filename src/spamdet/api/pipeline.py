from dataclasses import dataclass

from ..model.inference import PredictionResult, SpamClassifier
from ..outbreak.detector import OutbreakDetector, OutbreakResult
from ..preprocessing.homoglyphs import detect_mixed_script, strip_confusables
from ..preprocessing.url_tools import extract_urls
from ..preprocessing.zero_width import strip_zero_width
from ..subtype.ad_info_classifier import REKLAM
from ..subtype.detector import OTP, SubtypeDetector, SubtypeResult
from .config import REVIEW_CONFIDENCE_HIGH, REVIEW_CONFIDENCE_LOW, SPAM_TO_REKLAM_OVERRIDE_THRESHOLD


def needs_human_review(confidence: float) -> bool:
    return REVIEW_CONFIDENCE_LOW <= confidence <= REVIEW_CONFIDENCE_HIGH


def _should_override_spam_verdict(candidate: SubtypeResult) -> bool:
    """Whether a subtype-layer result is trustworthy enough to overturn
    the top-level model's own 'spam' call. otp is a deterministic rule -
    always trusted. reklam is the ML classifier - requires a distinctly
    higher bar than the routine reklam-vs-bilgilendirme split within
    already-'legitimate' text, since we're overriding the main model's
    own decision here, not just refining it. bilgilendirme is too weak a
    signal on its own to move a 'spam' verdict - it stays spam.
    """
    if candidate.subtype == OTP:
        return True
    if candidate.subtype == REKLAM and candidate.reklam_probability is not None:
        return candidate.reklam_probability >= SPAM_TO_REKLAM_OVERRIDE_THRESHOLD
    return False


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
    subtype: SubtypeResult | None


class ClassificationPipeline:
    """Ties Stage 1 preprocessing, the Stage 2 classifier, the Stage 2
    outbreak detector, and the legitimate-message subtype detector into
    the single per-message flow the API exposes.

    URL unshortening (built and tested in Stage 1) is intentionally NOT
    called synchronously here - resolving a redirect chain over the
    network can take seconds, which would make /classify's latency
    depend on an attacker-controlled server. Raw extracted URLs are
    still reported; unshortening is better suited to an async background
    enrichment step, noted in docs/production_readiness.md rather than
    built now.

    Messages the top-level model calls 'spam' also get a subtype check
    (not just 'legitimate' ones): the top-level fraud classifier was
    never retrained to separate "real ad" from "generic spam" within its
    own 'spam' bucket (see docs/subtype.md), so a confident subtype-layer
    'reklam'/'otp' result overrides 'spam' into 'legitimate' with that
    subtype attached - see _should_override_spam_verdict. Fraud-specific
    labels (phishing/gambling_scam/financial_urgency) are never touched
    by this - only the generic 'spam' bucket is reconsidered.
    """

    def __init__(
        self,
        classifier: SpamClassifier,
        outbreak_detector: OutbreakDetector | None = None,
        subtype_detector: SubtypeDetector | None = None,
    ):
        self.classifier = classifier
        self.outbreak_detector = outbreak_detector
        self.subtype_detector = subtype_detector

    def process(self, message_id: str, text: str) -> PipelineResult:
        cleaned = strip_zero_width(text)
        homoglyph_report = detect_mixed_script(cleaned)
        if homoglyph_report.is_mixed_script:
            cleaned = strip_confusables(cleaned)
        urls = extract_urls(cleaned)

        prediction = self.classifier.predict(cleaned)

        outbreak_result = None
        if self.outbreak_detector is not None:
            outbreak_result = self.outbreak_detector.ingest(message_id, text)

        # Subtype (otp/bilgilendirme/reklam) is only meaningful for
        # "legitimate" messages - but "spam" is checked too, since the
        # top-level model was never retrained to pull real ads out of its
        # own "spam" bucket. A confident otp/reklam result there
        # overrides "spam" into "legitimate". Fraud-specific labels are
        # never reconsidered.
        subtype_result = None
        if self.subtype_detector is not None:
            if prediction.label == "legitimate":
                subtype_result = self.subtype_detector.detect(cleaned)
            elif prediction.label == "spam":
                candidate = self.subtype_detector.detect(cleaned)
                if _should_override_spam_verdict(candidate):
                    subtype_result = candidate
                    # confidence reflects what actually drove this call
                    # (the override decision), not Stage A's stale,
                    # near-irrelevant original P(legitimate) - reporting
                    # that instead would print something like "legitimate
                    # (2% confidence)", which reads as broken even though
                    # it's technically an honest number from the wrong
                    # question. otp is a deterministic rule match
                    # (effectively certain); reklam uses the probability
                    # that actually triggered the override.
                    override_confidence = 0.95 if candidate.subtype == OTP else (candidate.reklam_probability or 0.0)
                    prediction = PredictionResult(
                        label="legitimate",
                        confidence=override_confidence,
                        probabilities=prediction.probabilities,
                    )

        return PipelineResult(
            message_id=message_id,
            text=text,
            cleaned_text=cleaned,
            prediction=prediction,
            homoglyph_attack_detected=homoglyph_report.is_mixed_script,
            urls_found=urls,
            needs_review=needs_human_review(prediction.confidence),
            outbreak=outbreak_result,
            subtype=subtype_result,
        )
