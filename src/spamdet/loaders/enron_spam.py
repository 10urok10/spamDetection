from pathlib import Path

from ..schema import Label, Lang, Record
from .base import find_column, read_csv_robust

SOURCE_NAME = "enron_spam"

TEXT_COLUMNS = ["text", "Text", "message", "Message", "body", "Body", "email"]
LABEL_COLUMNS = ["label", "Label", "class", "Class", "Category", "category"]

# "ham" doesn't tell us otp/reklam/bilgilendirme, so it maps to
# bilgilendirme as the general "not spam, informational" catch-all (see
# docs/model.md)
LABEL_MAP = {"spam": Label.SPAM, "ham": Label.BILGILENDIRME, "legitimate": Label.BILGILENDIRME}

FOLDER_LABEL_MAP = {"spam": Label.SPAM, "ham": Label.BILGILENDIRME}


def load(path: str | Path) -> list[Record]:
    """Load the Enron spam dataset, distributed either as a single CSV or
    as the original ham/spam/ folder-of-.txt-files layout."""
    p = Path(path)
    if p.is_dir():
        return _load_from_folders(p)
    return _load_from_csv(p)


def _load_from_folders(root: Path) -> list[Record]:
    records: list[Record] = []
    found_any_label_dir = False
    for label_dir in root.rglob("*"):
        if not label_dir.is_dir():
            continue
        label = FOLDER_LABEL_MAP.get(label_dir.name.strip().lower())
        if label is None:
            continue
        found_any_label_dir = True
        for txt_file in label_dir.glob("*.txt"):
            text = txt_file.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
            records.append(Record(text=text, label=label, source=SOURCE_NAME, lang=Lang.EN))
    if not found_any_label_dir:
        raise ValueError(
            f"[{SOURCE_NAME}] no 'ham'/'spam' subdirectories found under {root}; "
            "expected the original Enron folder layout or a CSV file, not a bare directory"
        )
    return records


def _load_from_csv(path: Path) -> list[Record]:
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
        records.append(Record(text=text, label=label, source=SOURCE_NAME, lang=Lang.EN))

    if unmapped:
        raise ValueError(f"[{SOURCE_NAME}] unmapped label values: {sorted(unmapped)}; extend LABEL_MAP")
    return records
