import pytest

from spamdet.loaders import turkish_sms_collection as loader
from spamdet.loaders.base import ColumnNotFoundError
from spamdet.schema import Label, Lang


def test_loads_records_with_standard_column_names(tmp_path):
    path = tmp_path / "sms.csv"
    path.write_text(
        "Message,Group\n"
        "Bedava bonus kazandiniz hemen tiklayin,spam\n"
        "Yarin saat 10da toplanti var,normal\n",
        encoding="utf-8",
    )
    records = loader.load(path)
    assert len(records) == 2
    assert records[0].label is Label.SPAM
    assert records[0].lang is Lang.TR
    assert records[0].source == "turkish_sms_collection"
    assert records[1].label is Label.LEGITIMATE


def test_handles_alternate_column_casing_and_naming(tmp_path):
    path = tmp_path / "sms2.csv"
    path.write_text("sms,etiket\nBonus kazandiniz,spam\n", encoding="utf-8")
    records = loader.load(path)
    assert len(records) == 1
    assert records[0].label is Label.SPAM


def test_raises_column_not_found_with_unrecognized_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("foo,bar\nx,y\n", encoding="utf-8")
    with pytest.raises(ColumnNotFoundError):
        loader.load(path)


def test_raises_on_unmapped_label_value(tmp_path):
    path = tmp_path / "unmapped.csv"
    path.write_text("Message,Group\nmerhaba,unknown_label\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unmapped label"):
        loader.load(path)


def test_skips_blank_text_rows(tmp_path):
    path = tmp_path / "blank.csv"
    path.write_text("Message,Group\n ,spam\nmerhaba naber,normal\n", encoding="utf-8")
    records = loader.load(path)
    assert len(records) == 1
    assert records[0].text == "merhaba naber"
