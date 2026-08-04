from pathlib import Path

import pandas as pd

from ..schema import Label, Lang, Record
from .base import ColumnNotFoundError, find_column, read_csv_robust

SOURCE_NAME = "sms_spam_collection"

# v1/v2 is the well-known raw UCI column naming that Kaggle mirrors
# frequently keep unchanged.
TEXT_COLUMNS = ["v2", "text", "Text", "message", "Message", "sms"]
LABEL_COLUMNS = ["v1", "label", "Label", "class", "Class", "Category", "category"]

LABEL_MAP = {"spam": Label.SPAM, "ham": Label.LEGITIMATE, "legitimate": Label.LEGITIMATE}


def load(path: str | Path) -> list[Record]:
    df = read_csv_robust(path)
    try:
        text_col = find_column(df, TEXT_COLUMNS, dataset_name=SOURCE_NAME)
        label_col = find_column(df, LABEL_COLUMNS, dataset_name=SOURCE_NAME)
    except ColumnNotFoundError:
        # Fall back to the original UCI distribution: headerless,
        # tab-separated, exactly two columns (label, text).
        df = read_csv_robust(path, sep="\t", header=None, names=["label", "text"])
        if df.shape[1] != 2:
            raise
        label_col, text_col = "label", "text"

    records: list[Record] = []
    unmapped: set[str] = set()
    for _, row in df.iterrows():
        raw_label = str(row[label_col]).strip().lower()
        label = LABEL_MAP.get(raw_label)
        if label is None:
            unmapped.add(raw_label)
            continue
        text = str(row[text_col]).strip()
        if not text or text.lower() == "nan":
            continue
        records.append(Record(text=text, label=label, source=SOURCE_NAME, lang=Lang.EN))

    if unmapped:
        raise ValueError(f"[{SOURCE_NAME}] unmapped label values: {sorted(unmapped)}; extend LABEL_MAP")
    return records
