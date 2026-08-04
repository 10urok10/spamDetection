import fakeredis
import pytest

from spamdet.api.pipeline import ClassificationPipeline, needs_human_review
from spamdet.model.inference import PredictionResult
from spamdet.outbreak.detector import OutbreakDetector


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
