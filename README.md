# spamdet

Turkish spam / fraud / gambling-scam / phishing detection MVP. Multi-stage
project; this README currently covers **Stage 1: data + preprocessing**.

## Stage 1 scope

- Ingest 5 public spam/ham datasets (3 Turkish, 2 English reference) into a
  single normalized schema (`text, label, source, lang`)
- Hand-authored + augmented + adversarial synthetic data for fraud
  subtypes with no public Turkish dataset (`gambling_scam`, `phishing`,
  `financial_urgency`), including deliberately spam-*looking* legitimate
  examples (real bank/cargo notifications) to guard against false
  positives
- Pre-processing modules: homoglyph/mixed-script normalizer, zero-width
  character cleaner, URL extractor + SSRF-safe unshortener

Model training, outbreak detection, and the FastAPI/dashboard layers are
later stages, not part of this codebase yet.

## Setup

Requires a dedicated conda environment (Python 3.11):

```
conda create -n spamdet python=3.11
conda activate spamdet
pip install -e ".[dev]"
```

Windows note: this project uses non-ASCII (Turkish) text throughout tests
and data. Set `PYTHONIOENCODING=utf-8` in your shell before running
anything if you see `UnicodeEncodeError` from printing - the default
Windows console codepage (e.g. cp1254) can't display all Unicode output.

## Running tests

```
pytest
```

All preprocessing and loader tests run fully offline (HTTP and DNS calls
are mocked). The one loader that makes a live network call
(`turkishsms_ds`, via Hugging Face `datasets`) is never exercised in
`tests/` - only from `scripts/build_dataset.py`.

## Building the merged dataset

1. Download the raw datasets described in `docs/datasets.md` and place them
   under `data/raw/<source_name>/` (gitignored - never commit raw data).
2. Optionally run `python scripts/inspect_raw.py` to print the actual
   columns found in each file, in case a loader needs its candidate column
   list extended.
3. Run:
   ```
   python scripts/build_dataset.py
   ```
   Add `--offline` to skip `turkishsms_ds` (no network access). Sources
   whose raw files aren't present yet are skipped with a notice rather than
   failing the whole build - the script works with however many of the 5
   sources you actually have.

Output: `data/processed/merged_dataset.csv` and `.parquet`, plus a
per-source/label/lang count summary printed to stdout.

## Generating synthetic data

```
python scripts/generate_synthetic.py
```

Reads `data/synthetic/seeds/*.yaml`, produces paraphrase-augmented and
adversarial (homoglyph + zero-width corrupted) variants, and writes
`data/synthetic/generated/{seeds,augmented,adversarial}.jsonl`. Fully
deterministic given `--rng-seed` (default 42) - no external API calls.

## Project layout

```
src/spamdet/
  schema.py              Label/Lang/Record - the normalized record schema
  loaders/                one module per public dataset + shared base.py
  preprocessing/
    homoglyphs.py          NFKD canonicalization + mixed-script detection
    zero_width.py          invisible-character stripping (+ injection for
                            adversarial generation)
    url_tools.py            URL extraction + SSRF-safe redirect unshortening
  synthetic/
    seeds.py                loads data/synthetic/seeds/*.yaml
    augment.py               rule-based paraphrase augmentation
    adversarial.py            homoglyph/zero-width adversarial variant generation
  merge.py                combines loader outputs into one deduplicated table
  build_dataset.py         scripts/build_dataset.py entry point
  generate_synthetic.py    scripts/generate_synthetic.py entry point
scripts/                  thin CLI wrappers + inspect_raw.py debug helper
data/
  raw/         gitignored - manually downloaded source files
  processed/   gitignored - merge.py output
  synthetic/
    seeds/       committed - hand-authored YAML seed examples
    generated/   gitignored - augment/adversarial script output
docs/
  datasets.md          per-source download/column/license notes
  licensing_notes.md   BerTurk-SpamSMS OpenRAIL-M warning - do not use it
```

## Known limitations (by design, documented not solved in Stage 1)

- **SSRF guard DNS-rebinding TOCTOU**: `url_tools.unshorten()` validates
  each redirect hop's resolved IP before connecting, but there's a small
  window between that check and `requests`' own connection. Full
  protection requires routing outbound requests through an isolated
  egress proxy/network - planned for Stage 3 (Dockerization); the
  `proxies` parameter on `unshorten()` exists so that can be wired in
  without changing call sites.
- **Kaggle downloads are manual**: no `kaggle` API credential handling in
  code by default (the `datafetch` extra installs the `kaggle` CLI
  separately if you want to script it yourself). Raw files are just
  expected to already exist under `data/raw/`.
- **Synthetic seed text is ASCII-only Turkish** (no `ç/ğ/ı/ö/ş/ü`
  diacritics) - deliberately, since a large share of real Turkish
  SMS/spam traffic is typed without them; diacritic handling itself is
  covered separately by the homoglyph/zero-width test suites.
- Rule-based paraphrasing (`TemplateParaphraser`) is intentionally simple
  and offline; an LLM-backed `Paraphraser` can be swapped in later without
  changing `augment_examples()`'s signature.
