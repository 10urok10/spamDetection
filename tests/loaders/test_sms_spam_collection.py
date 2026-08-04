from spamdet.loaders import sms_spam_collection as loader
from spamdet.schema import Label, Lang


def test_loads_records_with_v1_v2_columns(tmp_path):
    path = tmp_path / "spam.csv"
    path.write_text('v1,v2\nspam,"Free entry, win a prize now"\nham,"See you tomorrow"\n', encoding="utf-8")
    records = loader.load(path)
    assert len(records) == 2
    assert records[0].label is Label.SPAM
    assert records[0].lang is Lang.EN
    assert records[1].label is Label.LEGITIMATE


def test_falls_back_to_headerless_tab_separated_uci_format(tmp_path):
    path = tmp_path / "raw_uci.tsv"
    path.write_text("spam\tFree entry win a prize\nham\tSee you tomorrow\n", encoding="utf-8")
    records = loader.load(path)
    assert len(records) == 2
    assert records[0].label is Label.SPAM
    assert records[1].label is Label.LEGITIMATE
