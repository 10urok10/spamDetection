import fakeredis
import pytest

from spamdet.outbreak.detector import OutbreakDetector


@pytest.fixture
def detector():
    redis_client = fakeredis.FakeStrictRedis(decode_responses=False)
    return OutbreakDetector(redis_client, similarity_threshold=0.85)


def test_first_message_is_not_an_outbreak_candidate(detector):
    result = detector.ingest("msg-1", "Bonus kazandiniz hemen tiklayin bit.ly/x")
    assert result.is_outbreak_candidate is False
    assert result.similar_message_ids == []


def test_near_identical_second_message_flags_outbreak(detector):
    detector.ingest("msg-1", "Bonus kazandiniz hemen tiklayin bit.ly/x")
    result = detector.ingest("msg-2", "Bonus kazandiniz hemen tiklayin bit.ly/y")
    assert result.is_outbreak_candidate is True
    assert "msg-1" in result.similar_message_ids
    assert result.similarities["msg-1"] >= 0.85


def test_unrelated_second_message_does_not_flag_outbreak(detector):
    detector.ingest("msg-1", "Bonus kazandiniz hemen tiklayin bit.ly/x")
    result = detector.ingest("msg-2", "Yarin saat 15:00 te toplantimiz var")
    assert result.is_outbreak_candidate is False


def test_zero_width_corrupted_variant_still_detected_as_near_duplicate(detector):
    from spamdet.synthetic.adversarial import make_zero_width_variant
    import random

    original = "Hesabinizi dogrulamak icin linke tiklayin guvenlik-bankam.com"
    corrupted = make_zero_width_variant(original, fraction=0.5, rng=random.Random(1))

    detector.ingest("msg-1", original)
    result = detector.ingest("msg-2", corrupted)
    assert result.is_outbreak_candidate is True


def test_clean_text_strips_zero_width_characters():
    text = "bonus" + chr(0x200B) + "kazandiniz"
    assert OutbreakDetector.clean_text(text) == "bonuskazandiniz"


def test_third_similar_message_matches_both_prior_messages(detector):
    detector.ingest("msg-1", "Bonus kazandiniz hemen tiklayin bit.ly/x")
    detector.ingest("msg-2", "Bonus kazandiniz hemen tiklayin bit.ly/y")
    result = detector.ingest("msg-3", "Bonus kazandiniz hemen tiklayin bit.ly/z")
    assert set(result.similar_message_ids) >= {"msg-1", "msg-2"}
