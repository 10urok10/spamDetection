from pathlib import Path

import pandas as pd

from ..schema import Label, Lang, Record
from .base import ColumnNotFoundError, find_column, read_csv_robust

SOURCE_NAME = "turkish_sms_collection"

TEXT_COLUMNS = ["Message", "message", "text", "Text", "sms", "SMS", "Mesaj", "mesaj", "icerik", "İçerik"]
# GroupText ("Spam"/"Normal") takes priority over Group (numeric 1/2 codes,
# real meaning not self-evident) - real file has both, prefer the
# human-readable one so LABEL_MAP doesn't need dataset-version-specific
# numeric-code guesses.
LABEL_COLUMNS = [
    "GroupText",
    "grouptext",
    "Group",
    "group",
    "label",
    "Label",
    "Grup",
    "class",
    "Class",
    "Etiket",
    "etiket",
    "Durum",
]

LABEL_MAP = {
    "spam": Label.SPAM,
    # the public dataset only distinguishes spam/ham - "ham" doesn't tell
    # us otp/reklam/bilgilendirme, so it maps to bilgilendirme as the
    # general "not spam, informational" catch-all (see docs/model.md)
    "normal": Label.BILGILENDIRME,
    "legitimate": Label.BILGILENDIRME,
    "ham": Label.BILGILENDIRME,
}


def _read(path: str | Path) -> pd.DataFrame:
    """The real onurkarasoy/turkish-sms-collection export turned out to be
    semicolon-delimited (a common European-locale CSV convention), which
    the comma-delimited default either mis-parses into one column or, more
    often, trips the C parser's field-count check outright. Try comma
    first (in case some mirror really is comma-delimited), fall back to
    semicolon.
    """
    try:
        df = read_csv_robust(path)
        find_column(df, TEXT_COLUMNS, dataset_name=SOURCE_NAME)
        return df
    except (pd.errors.ParserError, ColumnNotFoundError):
        return read_csv_robust(path, sep=";")


def load(path: str | Path) -> list[Record]:
    df = _read(path)
    text_col = find_column(df, TEXT_COLUMNS, dataset_name=SOURCE_NAME)
    label_col = find_column(df, LABEL_COLUMNS, dataset_name=SOURCE_NAME)

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
        records.append(Record(text=text, label=label, source=SOURCE_NAME, lang=Lang.TR))

    if unmapped:
        raise ValueError(f"[{SOURCE_NAME}] unmapped label values: {sorted(unmapped)}; extend LABEL_MAP")
    return records
