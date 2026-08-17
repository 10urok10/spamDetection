import pandas as pd
import pytest

from spamdet.merge import merge_sources, write_processed
from spamdet.schema import Label, Lang, Record


def _rec(text, label=Label.BILGILENDIRME, source="src", lang=Lang.TR):
    return Record(text=text, label=label, source=source, lang=lang)


def test_merge_sources_unions_records_from_multiple_loaders():
    loaders = {
        "a": lambda: [_rec("merhaba"), _rec("nasilsin")],
        "b": lambda: [_rec("bonus kazandiniz", label=Label.SPAM)],
    }
    df = merge_sources(loaders)
    assert len(df) == 3
    assert set(df["text"]) == {"merhaba", "nasilsin", "bonus kazandiniz"}


def test_merge_sources_deduplicates_exact_text_lang_pairs():
    loaders = {
        "a": lambda: [_rec("ayni mesaj", source="a")],
        "b": lambda: [_rec("ayni mesaj", source="b")],
    }
    df = merge_sources(loaders)
    assert len(df) == 1
    assert df.iloc[0]["source"] == "a"  # first loader wins


def test_merge_sources_keeps_same_text_different_lang():
    loaders = {
        "a": lambda: [_rec("free money", lang=Lang.TR)],
        "b": lambda: [_rec("free money", lang=Lang.EN)],
    }
    df = merge_sources(loaders)
    assert len(df) == 2


def test_merge_sources_wraps_failing_loader_error_with_its_name():
    def _broken():
        raise ValueError("boom")

    loaders = {"a": lambda: [_rec("ok")], "broken_loader": _broken}
    with pytest.raises(RuntimeError, match="broken_loader"):
        merge_sources(loaders)


def test_write_processed_writes_csv_and_parquet(tmp_path):
    df = merge_sources({"a": lambda: [_rec("merhaba"), _rec("bonus", label=Label.SPAM)]})
    csv_path, parquet_path = write_processed(df, tmp_path)
    assert csv_path.exists()
    assert parquet_path.exists()

    reloaded = pd.read_csv(csv_path)
    assert len(reloaded) == 2
    assert set(reloaded["text"]) == {"merhaba", "bonus"}
