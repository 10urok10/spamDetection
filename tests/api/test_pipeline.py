import fakeredis
import pytest

from spamdet.api.pipeline import ClassificationPipeline, needs_human_review
from spamdet.model.inference import PredictionResult
from spamdet.outbreak.detector import OutbreakDetector
from spamdet.subtype.detector import SubtypeResult


class _FakeClassifier:
    def __init__(self, result: PredictionResult):
        self._result = result

    def predict(self, text: str) -> PredictionResult:
        return self._result


@pytest.mark.parametrize(
    "confidence,expected",
    [(0.39, False), (0.4, True), (0.5, True), (0.6, True), (0.61, False), (0.95, False)],
)
def test_needs_human_review_boundaries(confidence, expected):
    assert needs_human_review(confidence) is expected


def test_pipeline_cleans_zero_width_before_classifying():
    zw_text = "bo" + chr(0x200B) + "nus kazandiniz"
    fake_result = PredictionResult(label="gambling_scam", confidence=0.9, probabilities={"gambling_scam": 0.9})
    pipeline = ClassificationPipeline(_FakeClassifier(fake_result))

    result = pipeline.process("msg-1", zw_text)
    assert result.cleaned_text == "bonus kazandiniz"


def test_pipeline_strips_homoglyph_confusables():
    cyrillic_a = chr(0x0430)
    spoofed = "hes" + cyrillic_a + "bınızı doğrulayın"
    fake_result = PredictionResult(label="phishing", confidence=0.9, probabilities={"phishing": 0.9})
    pipeline = ClassificationPipeline(_FakeClassifier(fake_result))

    result = pipeline.process("msg-1", spoofed)
    assert result.homoglyph_attack_detected is True
    assert cyrillic_a not in result.cleaned_text


def test_pipeline_extracts_urls():
    fake_result = PredictionResult(label="phishing", confidence=0.9, probabilities={"phishing": 0.9})
    pipeline = ClassificationPipeline(_FakeClassifier(fake_result))

    result = pipeline.process("msg-1", "Tiklayin: https://ornek.com/dogrula hemen simdi")
    assert result.urls_found == ["https://ornek.com/dogrula"]


def test_pipeline_flags_needs_review_in_confidence_band():
    fake_result = PredictionResult(label="spam", confidence=0.5, probabilities={"spam": 0.5, "legitimate": 0.5})
    pipeline = ClassificationPipeline(_FakeClassifier(fake_result))

    result = pipeline.process("msg-1", "belirsiz bir mesaj")
    assert result.needs_review is True


def test_pipeline_does_not_flag_high_confidence():
    fake_result = PredictionResult(label="legitimate", confidence=0.95, probabilities={"legitimate": 0.95})
    pipeline = ClassificationPipeline(_FakeClassifier(fake_result))

    result = pipeline.process("msg-1", "yarin gorusuruz")
    assert result.needs_review is False


def test_pipeline_without_outbreak_detector_leaves_outbreak_none():
    fake_result = PredictionResult(label="spam", confidence=0.9, probabilities={"spam": 0.9})
    pipeline = ClassificationPipeline(_FakeClassifier(fake_result))
    result = pipeline.process("msg-1", "test")
    assert result.outbreak is None


def test_pipeline_with_outbreak_detector_flags_near_duplicates():
    fake_result = PredictionResult(label="gambling_scam", confidence=0.9, probabilities={"gambling_scam": 0.9})
    redis_client = fakeredis.FakeStrictRedis(decode_responses=False)
    outbreak_detector = OutbreakDetector(redis_client, similarity_threshold=0.85)
    pipeline = ClassificationPipeline(_FakeClassifier(fake_result), outbreak_detector)

    pipeline.process("msg-1", "Bonus kazandiniz hemen tiklayin bit.ly/x")
    result = pipeline.process("msg-2", "Bonus kazandiniz hemen tiklayin bit.ly/y")
    assert result.outbreak.is_outbreak_candidate is True
    assert "msg-1" in result.outbreak.similar_message_ids


def test_pipeline_without_subtype_detector_leaves_subtype_none():
    fake_result = PredictionResult(label="legitimate", confidence=0.9, probabilities={"legitimate": 0.9})
    pipeline = ClassificationPipeline(_FakeClassifier(fake_result))
    result = pipeline.process("msg-1", "Kargonuz dagitima cikmistir.")
    assert result.subtype is None


class _FakeSubtypeDetector:
    def __init__(self, result: SubtypeResult):
        self._result = result
        self.calls: list[str] = []

    def detect(self, text: str) -> SubtypeResult:
        self.calls.append(text)
        return self._result


def test_pipeline_never_checks_subtype_for_fraud_labels():
    fake_result = PredictionResult(label="phishing", confidence=0.9, probabilities={"phishing": 0.9})
    fake_subtype = _FakeSubtypeDetector(SubtypeResult(subtype="reklam", source="model", reklam_probability=0.99))
    pipeline = ClassificationPipeline(_FakeClassifier(fake_result), subtype_detector=fake_subtype)

    result = pipeline.process("msg-1", "test")
    assert result.subtype is None
    assert result.prediction.label == "phishing"
    assert fake_subtype.calls == []


def test_pipeline_spam_with_low_confidence_reklam_stays_spam():
    fake_result = PredictionResult(label="spam", confidence=0.9, probabilities={"spam": 0.9, "legitimate": 0.1})
    fake_subtype = _FakeSubtypeDetector(SubtypeResult(subtype="reklam", source="model", reklam_probability=0.35))
    pipeline = ClassificationPipeline(_FakeClassifier(fake_result), subtype_detector=fake_subtype)

    result = pipeline.process("msg-1", "test")
    assert result.prediction.label == "spam"
    assert result.subtype is None
    assert fake_subtype.calls == ["test"]  # checked, just not confident enough to override


def test_pipeline_spam_with_high_confidence_reklam_overrides_to_legitimate():
    fake_result = PredictionResult(label="spam", confidence=0.9, probabilities={"spam": 0.9, "legitimate": 0.1})
    fake_subtype = _FakeSubtypeDetector(SubtypeResult(subtype="reklam", source="model", reklam_probability=0.9))
    pipeline = ClassificationPipeline(_FakeClassifier(fake_result), subtype_detector=fake_subtype)

    result = pipeline.process("msg-1", "Bu hafta magazamizda yuzde 40 indirim var!")
    assert result.prediction.label == "legitimate"
    assert result.prediction.confidence == 0.1  # Stage A's own P(legitimate), not fabricated
    assert result.prediction.probabilities == {"spam": 0.9, "legitimate": 0.1}  # unchanged, for audit
    assert result.subtype.subtype == "reklam"
    assert result.subtype.source == "model"


def test_pipeline_spam_with_otp_match_always_overrides():
    fake_result = PredictionResult(label="spam", confidence=0.9, probabilities={"spam": 0.9, "legitimate": 0.1})
    fake_subtype = _FakeSubtypeDetector(SubtypeResult(subtype="otp", source="rule_otp", reklam_probability=None))
    pipeline = ClassificationPipeline(_FakeClassifier(fake_result), subtype_detector=fake_subtype)

    result = pipeline.process("msg-1", "Kodunuz: 123456. Kimseyle paylasmayin.")
    assert result.prediction.label == "legitimate"
    assert result.subtype.subtype == "otp"


def test_pipeline_spam_with_bilgilendirme_never_overrides():
    fake_result = PredictionResult(label="spam", confidence=0.9, probabilities={"spam": 0.9, "legitimate": 0.1})
    fake_subtype = _FakeSubtypeDetector(
        SubtypeResult(subtype="bilgilendirme", source="model", reklam_probability=0.05)
    )
    pipeline = ClassificationPipeline(_FakeClassifier(fake_result), subtype_detector=fake_subtype)

    result = pipeline.process("msg-1", "test")
    assert result.prediction.label == "spam"
    assert result.subtype is None


def test_pipeline_runs_subtype_detector_when_label_is_legitimate():
    fake_result = PredictionResult(label="legitimate", confidence=0.9, probabilities={"legitimate": 0.9})
    fake_subtype = _FakeSubtypeDetector(SubtypeResult(subtype="bilgilendirme", source="model", reklam_probability=0.1))
    pipeline = ClassificationPipeline(_FakeClassifier(fake_result), subtype_detector=fake_subtype)

    result = pipeline.process("msg-1", "Kargonuz dagitima cikmistir.")
    assert result.subtype.subtype == "bilgilendirme"
    assert fake_subtype.calls == ["Kargonuz dagitima cikmistir."]
