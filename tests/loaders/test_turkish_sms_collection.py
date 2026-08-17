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
    assert records[1].label is Label.BILGILENDIRME


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


def test_falls_back_to_semicolon_delimiter(tmp_path):
    # the real onurkarasoy/turkish-sms-collection export is semicolon-
    # delimited, which trips the comma-delimited C parser outright.
    path = tmp_path / "real_format.csv"
    path.write_text(
        "Message;Group;GroupText\r\n"
        "125 lira;2;Normal\r\n"
        "Bedava bonus kazandiniz, hemen tiklayin;1;Spam\r\n",
        encoding="utf-8",
    )
    records = loader.load(path)
    assert len(records) == 2
    assert records[0].label is Label.BILGILENDIRME
    assert records[1].label is Label.SPAM
    assert "bonus" in records[1].text.lower()


def test_prefers_grouptext_over_numeric_group_code(tmp_path):
    # Group is a numeric code (1/2) whose meaning isn't self-evident;
    # GroupText has the actual Spam/Normal label and must win.
    path = tmp_path / "grouptext.csv"
    path.write_text("Message;Group;GroupText\r\nmerhaba;2;Normal\r\n", encoding="utf-8")
    records = loader.load(path)
    assert len(records) == 1
    assert records[0].label is Label.BILGILENDIRME
