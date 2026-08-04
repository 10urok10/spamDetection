import json
from pathlib import Path
from typing import Callable

import pandas as pd

from .schema import Record

LoaderFn = Callable[[], list[Record]]


def records_to_dataframe(records: list[Record]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "text": r.text,
                "label": r.label.value,
                "source": r.source,
                "lang": r.lang.value,
                "extra": r.extra,
            }
            for r in records
        ],
        columns=["text", "label", "source", "lang", "extra"],
    )


def merge_sources(loaders: dict[str, LoaderFn]) -> pd.DataFrame:
    """Run each named loader, concatenate results, and drop exact
    duplicate (text, lang) pairs across sources. If a loader raises, the
    error is re-raised wrapped with the loader's name so a build failure
    points at which dataset broke.
    """
    all_records: list[Record] = []
    for name, loader_fn in loaders.items():
        try:
            all_records.extend(loader_fn())
        except Exception as exc:
            raise RuntimeError(f"loader {name!r} failed: {exc}") from exc

    df = records_to_dataframe(all_records)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["text", "lang"], keep="first").reset_index(drop=True)


def write_processed(df: pd.DataFrame, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = df.assign(extra=df["extra"].apply(json.dumps))

    csv_path = out_dir / "merged_dataset.csv"
    parquet_path = out_dir / "merged_dataset.parquet"
    serializable.to_csv(csv_path, index=False)
    serializable.to_parquet(parquet_path, index=False)
    return csv_path, parquet_path
