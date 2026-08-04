import re
from pathlib import Path
from typing import Sequence

import pandas as pd

# Older/Kaggle-mirrored Turkish CSV exports are frequently NOT UTF-8;
# cp1254 (Windows Turkish) and iso-8859-9 are common for pre-2015 academic
# exports, and the classic UCI SMS Spam Collection Kaggle mirror is
# famously latin-1.
ENCODING_FALLBACKS: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1254", "iso-8859-9", "latin1")


class ColumnNotFoundError(RuntimeError):
    def __init__(self, candidates: Sequence[str], available: Sequence[str], dataset_name: str):
        self.candidates = list(candidates)
        self.available = list(available)
        self.dataset_name = dataset_name
        super().__init__(
            f"[{dataset_name}] none of the candidate column names {list(candidates)!r} "
            f"were found. Actual columns in file: {list(available)!r}. "
            "Update this loader's candidate list to match the real file."
        )


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def find_column(df: pd.DataFrame, candidates: Sequence[str], *, dataset_name: str) -> str:
    """Case/whitespace/punctuation-insensitive column lookup. Raises
    ColumnNotFoundError listing the actual columns present when no
    candidate matches, so a loader fails loudly instead of silently
    reading the wrong column.
    """
    lookup = {_normalize_name(c): c for c in df.columns}
    for candidate in candidates:
        normalized = _normalize_name(candidate)
        if normalized in lookup:
            return lookup[normalized]
    raise ColumnNotFoundError(candidates, list(df.columns), dataset_name)


def read_csv_robust(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a CSV trying a sequence of encodings, since public Turkish/UCI
    dataset mirrors are inconsistent about UTF-8. Raises the last decoding
    error if none of the fallback encodings work.
    """
    last_error: Exception | None = None
    for encoding in ENCODING_FALLBACKS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
    assert last_error is not None
    raise last_error
