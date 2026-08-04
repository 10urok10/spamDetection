import pytest
from pydantic import ValidationError

from spamdet.schema import Label, Lang, Record


def test_valid_record_construction():
    rec = Record(text="Merhaba, nasılsın?", label=Label.LEGITIMATE, source="synthetic_seed", lang=Lang.TR)
    assert rec.text == "Merhaba, nasılsın?"
    assert rec.label is Label.LEGITIMATE
    assert rec.lang is Lang.TR
    assert rec.extra == {}


def test_blank_text_raises():
    with pytest.raises(ValidationError):
        Record(text="   ", label=Label.SPAM, source="x", lang=Lang.TR)


def test_unknown_label_raises():
    with pytest.raises(ValidationError):
        Record(text="metin", label="not_a_real_label", source="x", lang=Lang.TR)


def test_text_is_stripped():
    rec = Record(text="  merhaba  ", label=Label.LEGITIMATE, source="x", lang=Lang.TR)
    assert rec.text == "merhaba"


@pytest.mark.parametrize(
    "label,expected_coarse",
    [
        (Label.LEGITIMATE, Label.LEGITIMATE),
        (Label.SPAM, Label.SPAM),
        (Label.GAMBLING_SCAM, Label.SPAM),
        (Label.PHISHING, Label.SPAM),
        (Label.FINANCIAL_URGENCY, Label.SPAM),
    ],
)
def test_coarse_collapses_non_legitimate_to_spam(label, expected_coarse):
    assert label.coarse is expected_coarse


def test_extra_field_accepts_passthrough_metadata():
    rec = Record(
        text="test",
        label=Label.PHISHING,
        source="synthetic_adversarial",
        lang=Lang.TR,
        extra={"variant_type": "homoglyph", "original_text": "test"},
    )
    assert rec.extra["variant_type"] == "homoglyph"
