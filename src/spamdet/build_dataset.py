import argparse
from pathlib import Path
from typing import Callable

from .loaders import (
    enron_spam,
    sms_spam_collection,
    turkish_sms_collection,
    turkish_spam_dataset,
    turkishsms_ds,
)
from .merge import merge_sources, write_processed
from .schema import Record

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed"

LoaderFn = Callable[[], list[Record]]


def _find_first_csv(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    csvs = sorted(directory.glob("*.csv"))
    return csvs[0] if csvs else None


def build_loaders(raw_dir: Path, *, include_turkishsms_ds: bool = True) -> dict[str, LoaderFn]:
    """Wire up one loader per source dataset found under ``raw_dir``.
    Sources whose raw files haven't been downloaded yet are skipped with
    a printed notice rather than failing the whole build - Stage 1 must
    still work with only some datasets present.
    """
    loaders: dict[str, LoaderFn] = {}

    tsc_csv = _find_first_csv(raw_dir / "turkish_sms_collection")
    if tsc_csv:
        loaders["turkish_sms_collection"] = lambda p=tsc_csv: turkish_sms_collection.load(p)
    else:
        print(f"[skip] turkish_sms_collection: no CSV found under {raw_dir / 'turkish_sms_collection'}")

    tsd_csv = _find_first_csv(raw_dir / "turkish_spam_dataset")
    if tsd_csv:
        loaders["turkish_spam_dataset"] = lambda p=tsd_csv: turkish_spam_dataset.load(p)
    else:
        print(f"[skip] turkish_spam_dataset: no CSV found under {raw_dir / 'turkish_spam_dataset'}")

    ssc_csv = _find_first_csv(raw_dir / "sms_spam_collection")
    if ssc_csv:
        loaders["sms_spam_collection"] = lambda p=ssc_csv: sms_spam_collection.load(p)
    else:
        print(f"[skip] sms_spam_collection: no CSV found under {raw_dir / 'sms_spam_collection'}")

    enron_dir = raw_dir / "enron_spam"
    if enron_dir.is_dir():

        def _load_enron(d: Path = enron_dir) -> list[Record]:
            try:
                return enron_spam.load(d)
            except ValueError:
                csv_path = _find_first_csv(d)
                if csv_path is None:
                    raise
                return enron_spam.load(csv_path)

        loaders["enron_spam"] = _load_enron
    else:
        print(f"[skip] enron_spam: directory not found at {enron_dir}")

    if include_turkishsms_ds:
        loaders["turkishsms_ds"] = turkishsms_ds.load
    else:
        print("[skip] turkishsms_ds: --offline flag set")

    return loaders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the merged Stage 1 dataset from raw sources.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip turkishsms_ds, which requires network access to Hugging Face",
    )
    args = parser.parse_args(argv)

    loaders = build_loaders(args.raw_dir, include_turkishsms_ds=not args.offline)
    if not loaders:
        print("No datasets available to load. Place raw files under data/raw/<source>/ first. See docs/datasets.md.")
        return 1

    df = merge_sources(loaders)
    if df.empty:
        print("All configured loaders returned zero records.")
        return 1

    csv_path, parquet_path = write_processed(df, args.out_dir)
    print(f"Wrote {len(df)} records to:\n  {csv_path}\n  {parquet_path}")
    print("\nCounts by source:")
    print(df.groupby("source").size().to_string())
    print("\nCounts by label:")
    print(df.groupby("label").size().to_string())
    print("\nCounts by lang:")
    print(df.groupby("lang").size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
