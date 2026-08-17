from typing import Callable

from ..schema import Label, Lang, Record

SOURCE_NAME = "turkishsms_ds"
HF_REPO = "akuysal/turkishSMS-ds"

# Confirmed schema (2026-08): columns `text`, `label` with string values
# "legitimate"/"spam"; also has `sms_length` (unused here). "legitimate"
# doesn't tell us otp/reklam/bilgilendirme, so it maps to bilgilendirme
# as the general "not spam, informational" catch-all (see docs/model.md).
LABEL_MAP = {"legitimate": Label.BILGILENDIRME, "spam": Label.SPAM}


def load(*, revision: str | None = None, dataset_loader: Callable[..., object] | None = None) -> list[Record]:
    """Load akuysal/turkishSMS-ds from Hugging Face.

    ``dataset_loader`` defaults to ``datasets.load_dataset`` and is
    injectable so tests can supply an in-memory fake without a network
    call. This function performs a live network call in its default
    configuration - only call it from build scripts, never from tests.
    """
    if dataset_loader is None:
        from datasets import load_dataset as dataset_loader  # type: ignore[assignment]

    ds = dataset_loader(HF_REPO, revision=revision)

    records: list[Record] = []
    unmapped: set[str] = set()
    for split in ds.values():
        for row in split:
            raw_label = str(row["label"]).strip().lower()
            label = LABEL_MAP.get(raw_label)
            if label is None:
                unmapped.add(raw_label)
                continue
            text = str(row["text"]).strip()
            if not text:
                continue
            records.append(Record(text=text, label=label, source=SOURCE_NAME, lang=Lang.TR))

    if unmapped:
        raise ValueError(f"[{SOURCE_NAME}] unmapped label values: {sorted(unmapped)}; extend LABEL_MAP")
    return records
