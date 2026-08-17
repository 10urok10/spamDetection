import pytest
from datasets import Dataset, DatasetDict

from spamdet.loaders import turkishsms_ds as loader
from spamdet.schema import Label, Lang


def _fake_dataset_loader(rows: list[dict]):
    """Builds an in-memory DatasetDict, standing in for
    ``datasets.load_dataset`` - no network call involved."""

    def _loader(repo: str, revision: str | None = None):
        return DatasetDict({"train": Dataset.from_list(rows)})

    return _loader


def test_loads_records_and_maps_labels_no_network_call():
    rows = [
        {"text": "Bedava bonus kazandiniz", "label": "spam", "sms_length": 24},
        {"text": "Yarin gorusuruz", "label": "legitimate", "sms_length": 15},
    ]
    records = loader.load(dataset_loader=_fake_dataset_loader(rows))
    assert len(records) == 2
    assert records[0].label is Label.SPAM
    assert records[0].lang is Lang.TR
    assert records[0].source == "turkishsms_ds"
    assert records[1].label is Label.BILGILENDIRME


def test_raises_on_unmapped_label_value():
    rows = [{"text": "merhaba", "label": "unknown", "sms_length": 7}]
    with pytest.raises(ValueError, match="unmapped label"):
        loader.load(dataset_loader=_fake_dataset_loader(rows))


def test_skips_blank_text_rows():
    rows = [
        {"text": "  ", "label": "spam", "sms_length": 0},
        {"text": "gercek mesaj", "label": "legitimate", "sms_length": 12},
    ]
    records = loader.load(dataset_loader=_fake_dataset_loader(rows))
    assert len(records) == 1
    assert records[0].text == "gercek mesaj"
