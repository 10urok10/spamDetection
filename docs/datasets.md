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
- Expected columns: text-like (`Message`/`text`/`sms`/...), label-like
  (`Group`/`label`/`class`/...) - see `loaders/turkish_sms_collection.py`
  for the full candidate list
- License/attribution: check the Kaggle dataset page at download time and
  record the license here once confirmed

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
- Place at: `data/raw/turkish_spam_dataset/<file>.csv`
- Expected columns: text-like (`text`/`email`/`message`/...), label-like
  (`label`/`Category`/`class`/...) - see
  `loaders/turkish_spam_dataset.py`
- License/attribution: check the Kaggle dataset page at download time

## sms_spam_collection

- Source: UCI Machine Learning Repository / common Kaggle mirrors
- Language: English, binary spam/ham - used as a reference/comparison set,
  not for Turkish-specific evaluation
- Place at: `data/raw/sms_spam_collection/<file>.csv`
- Handles both the common Kaggle mirror column naming (`v1`/`v2`) and the
  original headerless tab-separated UCI distribution as a fallback

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
