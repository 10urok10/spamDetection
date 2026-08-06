# Data sources (Stage 1)

Raw files are never committed to the repo (`data/raw/` is gitignored). Download
each dataset manually and place it under `data/raw/<source_name>/` as
described below, then run `scripts/build_dataset.py`.

Column names below are *expected candidates* configured in each loader
(`src/spamdet/loaders/*.py`) - Kaggle dataset pages are JS-rendered and the
exact columns can't be confirmed without downloading the file. If a loader
raises `ColumnNotFoundError`, run `scripts/inspect_raw.py` to print the
actual columns found and extend the loader's candidate list to match.

## turkish_sms_collection

- Source: Kaggle, `onurkarasoy/turkish-sms-collection`
- Language: Turkish, binary spam/normal
- Place at: `data/raw/turkish_sms_collection/<file>.csv` (any CSV in that
  directory is picked up)
- **Downloaded and confirmed (2026-08)**: `TurkishSMSCollection.csv`,
  **semicolon-delimited** (not comma), CRLF line endings, columns
  `Message;Group;GroupText` - `Group` is a numeric code (1/2) whose
  meaning isn't self-evident, `GroupText` is the human-readable
  `Spam`/`Normal` label the loader actually uses (prioritized over
  `Group` in `LABEL_COLUMNS`). 4,737 rows loaded via `kaggle datasets
  download`.
- License/attribution: Kaggle CLI reports `License(s): unknown` - not
  independently re-verified beyond that; treat as unresolved before any
  wider distribution.

## turkishsms_ds

- Source: Hugging Face, `akuysal/turkishSMS-ds`
- Language: Turkish, binary `legitimate`/`spam`
- Loaded via `datasets.load_dataset` - no manual download needed, but this
  loader makes a live network call (only invoke from `build_dataset.py`,
  never from tests)
- Confirmed columns (2026-08): `text`, `label`, `sms_length`; 850 rows
  (765 train / 85 validation)
- Citation: Uysal et al. 2013, *Elektronika ir Elektrotechnika*

## turkish_spam_dataset

- Source: Kaggle, `cuneytdemir/turkish-spam-dataset`
- Language: Turkish, binary spam/legitimate email
- Place at: `data/raw/turkish_spam_dataset/<file>.csv` or `.xlsx`
- **Downloaded and confirmed (2026-08)**: the download contains both
  `trspam.csv` and `trspam.xlsx`. **The `.csv` is malformed** - unescaped
  multi-line quoted email bodies break the C parser (`Error tokenizing
  data`). The `.xlsx` is clean but has no real header row (first data row
  reads as columns) and one trailing all-blank footer row. The loader
  prefers `.xlsx` when present (`build_dataset.py`'s `_find_first_file`)
  and reads it by column position, dropping the blank-label footer row.
  752 rows loaded (496 ham / ~329 spam before cleanup). Some rows contain
  raw base64-encoded HTML email bodies (undecoded MIME content) - a
  source-data quality issue left as-is, not decoded.
- License/attribution: Kaggle CLI reports `License(s): unknown`.

## sms_spam_collection

- Source: UCI Machine Learning Repository / common Kaggle mirrors
- Language: English, binary spam/ham - used as a reference/comparison set,
  not for Turkish-specific evaluation
- Place at: `data/raw/sms_spam_collection/<file>.csv`
- Handles both the common Kaggle mirror column naming (`v1`/`v2`) and the
  original headerless tab-separated UCI distribution as a fallback
- **Downloaded and confirmed (2026-08)**: used the `uciml/sms-spam-collection-dataset`
  Kaggle mirror (`spam.csv`, `v1`/`v2` columns as expected, 5,158 rows
  after cleanup). Kaggle CLI reports `License(s): unknown` for this mirror
  too - the original UCI/SMS Spam Collection dataset itself is commonly
  cited as available for research use; verify the specific terms before
  wider distribution.

## enron_spam

- Source: Enron spam dataset (various public mirrors)
- Language: English, reference/comparison set
- Place at: `data/raw/enron_spam/` - either:
  - the original layout with `ham/` and `spam/` subfolders of `.txt` files, or
  - a single CSV file inside that directory
- `scripts/build_dataset.py` tries the folder layout first and falls back
  to any CSV found in the directory

## Label mapping note

All five public datasets only distinguish spam vs. ham/legitimate. They map
to `Label.SPAM` / `Label.LEGITIMATE` in our schema. The fine-grained fraud
subtypes (`gambling_scam`, `phishing`, `financial_urgency`) come exclusively
from our own synthetic data (`data/synthetic/seeds/`) - no public Turkish
dataset for those categories exists. `Label.coarse` collapses everything
non-legitimate to `spam` so both label granularities can be trained on
together in Stage 2.

See `docs/licensing_notes.md` for the one hard licensing constraint that
affects this project (not a dataset, a pretrained model).
