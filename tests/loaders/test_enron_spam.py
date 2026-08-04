from spamdet.loaders import enron_spam as loader
from spamdet.schema import Label, Lang


def test_loads_from_csv(tmp_path):
    path = tmp_path / "enron.csv"
    path.write_text(
        "text,label\n" '"Buy cheap meds now",spam\n' '"Meeting moved to 3pm",ham\n',
        encoding="utf-8",
    )
    records = loader.load(path)
    assert len(records) == 2
    assert records[0].label is Label.SPAM
    assert records[0].lang is Lang.EN
    assert records[1].label is Label.LEGITIMATE


def test_loads_from_ham_spam_folder_layout(tmp_path):
    spam_dir = tmp_path / "spam"
    ham_dir = tmp_path / "ham"
    spam_dir.mkdir()
    ham_dir.mkdir()
    (spam_dir / "0001.txt").write_text("Buy cheap meds now", encoding="utf-8")
    (ham_dir / "0001.txt").write_text("Meeting moved to 3pm", encoding="utf-8")

    records = loader.load(tmp_path)
    labels = {r.text: r.label for r in records}
    assert labels["Buy cheap meds now"] is Label.SPAM
    assert labels["Meeting moved to 3pm"] is Label.LEGITIMATE


def test_raises_on_directory_without_ham_spam_subfolders(tmp_path):
    (tmp_path / "misc").mkdir()
    import pytest

    with pytest.raises(ValueError, match="ham.*spam"):
        loader.load(tmp_path)
