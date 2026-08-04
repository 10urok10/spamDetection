from pathlib import Path

from ..schema import Label, Lang, Record
from .base import find_column, read_csv_robust

SOURCE_NAME = "turkish_spam_dataset"

TEXT_COLUMNS = ["text", "Text", "email", "Email", "message", "Message", "icerik", "İçerik", "mesaj", "Mesaj"]
LABEL_COLUMNS = ["label", "Label", "Category", "category", "class", "Class", "etiket", "Etiket", "spam"]

LABEL_MAP = {
    "spam": Label.SPAM,
    "ham": Label.LEGITIMATE,
    "normal": Label.LEGITIMATE,
    "legitimate": Label.LEGITIMATE,
    "1": Label.SPAM,
    "0": Label.LEGITIMATE,
}


def load(path: str | Path) -> list[Record]:
    df = read_csv_robust(path)
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
