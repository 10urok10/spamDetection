import pytest

from spamdet.loaders import turkish_spam_dataset as loader
from spamdet.loaders.base import ColumnNotFoundError
from spamdet.schema import Label, Lang


def test_loads_records_with_standard_columns(tmp_path):
    path = tmp_path / "email.csv"
    path.write_text(
        "text,label\n"
        "Hesabiniza 5000 TL bonus tanimlandi tiklayin,spam\n"
        "Toplanti saati degisti,legitimate\n",
        encoding="utf-8",
    )
    records = loader.load(path)
    assert len(records) == 2
    assert records[0].label is Label.SPAM
    assert records[0].lang is Lang.TR
    assert records[1].label is Label.LEGITIMATE


def test_handles_numeric_label_encoding(tmp_path):
    path = tmp_path / "numeric.csv"
    path.write_text("email,Category\nbedava kazan,1\nnormal mesaj,0\n", encoding="utf-8")
    records = loader.load(path)
    assert records[0].label is Label.SPAM
    assert records[1].label is Label.LEGITIMATE


def test_raises_column_not_found(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ColumnNotFoundError):
        loader.load(path)
