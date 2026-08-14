from unittest.mock import MagicMock

from spamdet.subtype.ad_info_classifier import AdInfoPrediction, BILGILENDIRME, REKLAM
from spamdet.subtype.detector import SubtypeDetector


def test_otp_rule_short_circuits_before_ml_classifier():
    fake_ad_info = MagicMock()
    detector = SubtypeDetector(fake_ad_info)

    result = detector.detect("Tek kullanimlik sifreniz: 482910. Bu kodu kimseyle paylasmayin.")

    assert result.subtype == "otp"
    assert result.source == "rule_otp"
    assert result.reklam_probability is None
    fake_ad_info.predict.assert_not_called()


def test_falls_back_to_ml_classifier_when_not_otp():
    fake_ad_info = MagicMock()
    fake_ad_info.predict.return_value = AdInfoPrediction(subtype=REKLAM, confidence=0.8, reklam_probability=0.8)
    detector = SubtypeDetector(fake_ad_info)

    result = detector.detect("Bu hafta magazamizda yuzde 30 indirim var, kacirmayin!")

    assert result.subtype == REKLAM
    assert result.source == "model"
    assert result.reklam_probability == 0.8
    fake_ad_info.predict.assert_called_once()


def test_ml_classifier_can_return_bilgilendirme():
    fake_ad_info = MagicMock()
    fake_ad_info.predict.return_value = AdInfoPrediction(
        subtype=BILGILENDIRME, confidence=0.9, reklam_probability=0.1
    )
    detector = SubtypeDetector(fake_ad_info)

    result = detector.detect("Kargonuz dagitima cikmistir.")

    assert result.subtype == BILGILENDIRME
    assert result.source == "model"
