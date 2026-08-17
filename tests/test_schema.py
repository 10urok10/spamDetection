import pytest
from pydantic import ValidationError

from spamdet.schema import Label, Lang, Record


def test_valid_record_construction():
    rec = Record(text="Merhaba, nasılsın?", label=Label.BILGILENDIRME, source="synthetic_seed", lang=Lang.TR)
    assert rec.text == "Merhaba, nasılsın?"
    assert rec.label is Label.BILGILENDIRME
    assert rec.lang is Lang.TR
    assert rec.extra == {}


def test_blank_text_raises():
    with pytest.raises(ValidationError):
        Record(text="   ", label=Label.SPAM, source="x", lang=Lang.TR)


def test_unknown_label_raises():
    with pytest.raises(ValidationError):
        Record(text="metin", label="not_a_real_label", source="x", lang=Lang.TR)


def test_text_is_stripped():
    rec = Record(text="  merhaba  ", label=Label.BILGILENDIRME, source="x", lang=Lang.TR)
    assert rec.text == "merhaba"


def test_extra_field_accepts_passthrough_metadata():
    rec = Record(
        text="test",
        label=Label.SPAM,
        source="synthetic_adversarial",
        lang=Lang.TR,
        extra={"variant_type": "homoglyph", "original_text": "test"},
    )
    assert rec.extra["variant_type"] == "homoglyph"
