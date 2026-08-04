"""Throwaway helper: print columns + a few sample rows for every CSV under
data/raw/, and file counts for any ham/spam folder layout (Enron). Run this
after downloading raw datasets to calibrate each loader's candidate column
names in src/spamdet/loaders/*.py - real column names aren't known until
the files are actually downloaded.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spamdet.loaders.base import read_csv_robust  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"


def inspect_csv(path: Path) -> None:
    print(f"\n=== {path.relative_to(RAW_DIR)} ===")
    try:
        df = read_csv_robust(path)
    except Exception as exc:
        print(f"  failed to read: {exc}")
        return
    print(f"  columns: {list(df.columns)}")
    print(f"  rows: {len(df)}")
    with_option = df.head(3).to_string(max_colwidth=60)
    print("\n".join(f"  {line}" for line in with_option.splitlines()))


def inspect_folder_layout(path: Path) -> None:
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    if not subdirs:
        return
    print(f"\n=== {path.relative_to(RAW_DIR)} (folder layout) ===")
    for d in subdirs:
        count = sum(1 for _ in d.glob("*.txt"))
        print(f"  {d.name}/: {count} .txt files")


def main() -> int:
    if not RAW_DIR.is_dir():
        print(f"{RAW_DIR} does not exist")
        return 1

    found_anything = False
    for entry in sorted(RAW_DIR.iterdir()):
        if entry.is_file() and entry.suffix.lower() == ".csv":
            inspect_csv(entry)
            found_anything = True
        elif entry.is_dir():
            csvs = sorted(entry.glob("*.csv"))
            if csvs:
                for csv_path in csvs:
                    inspect_csv(csv_path)
                    found_anything = True
            else:
                inspect_folder_layout(entry)
                found_anything = True

    if not found_anything:
        print(f"No raw dataset files found under {RAW_DIR}. See docs/datasets.md for download instructions.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
