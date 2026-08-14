# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Windows machine. Bare `python`/`py` are NOT on PATH (resolve to a Windows Store
stub). Use the dedicated conda env instead:

```
conda activate spamdet
# or, without activating:
C:\Users\<user>\anaconda3\envs\spamdet\python.exe -m pytest
```

The env has four installable extras layered on the base `pip install -e .`
(Stage 1 only): `dev` (pytest/fakeredis/responses), `train` (torch/
transformers/onnxruntime/optimum — Stage 2), `outbreak` (redis client —
Stage 2), `api` (fastapi/uvicorn/prometheus-client — Stage 3). Install what
you need: `pip install -e ".[dev,train,outbreak,api]"` for everything.

Set `PYTHONIOENCODING=utf-8` before running anything — tests and data are
full of Turkish text and the default Windows console codepage (cp1254)
can't print it, producing spurious `UnicodeEncodeError`s unrelated to
actual bugs.

GPU training on a Blackwell card (RTX 50-series) needs a CUDA wheel index
matched to the driver — plain `pip install torch` silently resolves to a
CPU build or one too old for `sm_120`. Check `nvidia-smi` for the driver's
CUDA version and install e.g. `pip install torch==2.13.0+cu132
--index-url https://download.pytorch.org/whl/cu132`. The `train` extra's
comment in `pyproject.toml` has the full explanation.

Docker/WSL2 on this machine has been observed to become unresponsive or
resource-heavy under sustained use. For local API testing that doesn't
need real Redis persistence, run against an in-memory `fakeredis` client
via `spamdet.api.app.create_app(classifier=..., redis_client=...)` instead
of `docker compose up` — same code path, no Docker dependency.

## Commands

```
pytest                                    # full suite, offline, ~10s
pytest tests/preprocessing/               # one module
pytest tests/api/test_app.py::test_health # one test
pytest -q --cov=spamdet --cov-report=term-missing   # with coverage

python scripts/build_dataset.py [--offline]      # merge raw datasets -> data/processed/
python scripts/generate_synthetic.py             # seeds -> augmented -> adversarial jsonl
python scripts/train_model.py [--offline]        # fine-tune -> models/spamdet-mdeberta/
python scripts/export_onnx.py                    # -> models/spamdet-mdeberta-onnx/
python scripts/run_api.py                        # serve on :8000 (needs Redis reachable)
python scripts/inspect_raw.py                    # print real columns of files in data/raw/

docker compose up -d redis      # Redis only, for local dev against scripts/run_api.py
docker compose up -d --build    # full api+redis stack (needs models/ present, mounted read-only)
```

Almost everything defaults to skipping steps gracefully rather than
failing: `build_dataset.py`/`train.py` print `[skip] <source>: ...` for
any of the 5 public datasets not present under `data/raw/`, and still
produce output from whatever sources *are* available. `--offline` skips
only `turkishsms_ds`, the one loader that makes a live network call
(Hugging Face `datasets`) — it's therefore also the one loader never
exercised by `pytest`.

## Architecture

**Three-stage pipeline, each stage's code lives in its own `src/spamdet/`
subpackage and can be exercised independently:**

1. **Data + preprocessing** (`loaders/`, `preprocessing/`, `synthetic/`,
   `merge.py`) — five public spam/ham datasets get parsed into one
   `Record` schema (`schema.py`: `text, label, source, lang, extra`) by
   per-source loaders that fail loudly (`ColumnNotFoundError` listing
   actual columns) rather than silently misreading a format. Real
   downloaded files have turned out to need format-specific handling not
   guessable in advance — semicolon-delimited CSVs, `.xlsx` siblings
   preferred over malformed `.csv` exports, etc. — see `docs/datasets.md`
   for what was actually found in each download. `preprocessing/`
   (homoglyph/mixed-script detection, zero-width stripping, SSRF-safe URL
   unshortening) is deliberately Turkish-diacritic-safe: it must not
   flag/mangle `ç ğ ı ö ş ü İ` while still catching real evasion — see the
   test suites for the specific false-positive guards. `synthetic/`
   generates the fraud-subtype training data no public Turkish dataset
   has (`gambling_scam`, `phishing`, `financial_urgency`): hand-written
   seeds in `data/synthetic/seeds/*.yaml` → rule-based paraphrase
   augmentation → homoglyph/zero-width adversarial corruption, all
   offline and deterministic given `--rng-seed`.

2. **Model + outbreak detection** (`model/`, `outbreak/`) —
   `model/dataset.py` combines the Stage-1 loaders' output with the
   synthetic data and does a stratified train/val/test split;
   `model/train.py` fine-tunes mDeBERTa-v3-base with focal loss
   (`focal_loss.py`) for the class imbalance; `model/export_onnx.py`
   produces an ONNX build. `model/inference.py`'s `SpamClassifier`
   transparently loads either backend — it picks ONNX if `model.onnx`
   exists in the given directory, else falls back to the PyTorch
   checkpoint — so callers never need to know which one is deployed.
   `outbreak/` is a from-scratch SimHash (64-bit, stdlib-only) + Redis
   LSH-banding near-duplicate detector for catching text-spun mass
   blasts; `docs/outbreak.md` explains why banding intentionally favors
   more/narrower bands (recall over precision — a second exact-similarity
   check downstream corrects false candidates, but a missed candidate is
   unrecoverable).

3. **API gateway** (`api/`) — `api/pipeline.py`'s `ClassificationPipeline`
   is the one place that wires Stage 1 preprocessing + Stage 2 classifier
   + Stage 2 outbreak detector into the actual per-message flow
   `app.py`'s `/classify` route calls. `api/app.py`'s `create_app()` is a
   factory (not a module-level `app` object) specifically so tests and
   ad-hoc scripts can inject a fake classifier/Redis client instead of
   the real ones the lifespan handler builds from `api/config.py`'s
   env-var lookups.

**Label taxonomy and what `spam` means**: five top-level labels —
`legitimate, spam, gambling_scam, phishing, financial_urgency`
(`model/labels.py`, fixed order = fixed integer ids everywhere). The
public datasets are binary (spam/ham) and only ever produce
`legitimate`/`spam`; the three fraud subtypes exist exclusively in the
synthetic seed data. There is no synthetic `spam` seed file — that label
only ever comes from the public datasets. Fraud intent specifically is
what the other three labels are for.

**Advertising messages are `legitimate`, not `spam`** — this reversed
partway through the project (an earlier round had them as `spam`; see
git history on `docs/model.md` for the "hepsiburada kuponu" example that
prompted the reversal). A real, identifiable, regulated marketing SMS
(Mersis number, opt-out mechanism) is a `legitimate` message whose
*subtype* is `reklam` — see the subtype system below. Don't move ad-like
text back to top-level `spam` without re-reading why.

**Legitimate-message subtypes** (`subtype/`, separate from the five
top-level labels above): an SMS-operator compliance layer that further
splits whatever the top-level model calls `legitimate` into `otp` /
`bilgilendirme` (informational) / `reklam` (advertisement) — built so ad
content can be checked against the right (separately KVKK-consented)
sending channel. `otp` is a deterministic regex rule
(`subtype/rules.py`); `reklam` vs `bilgilendirme` is a lightweight
TF-IDF + logistic-regression classifier (`subtype/ad_info_classifier.py`),
deliberately not another transformer fine-tune — evaluate the cheap
option first, escalate only if it underperforms (see `docs/subtype.md`).
**No Mersis-number/opt-out-phrase rule exists for `reklam`** — a real
user-supplied customer-satisfaction-survey message contains both markers
without being an ad, disproving that shortcut; those signals are ML
features, not a hard-coded trigger. `spam`/fraud-subtype messages never
get a subtype — only `legitimate` ones do. This whole layer runs
strictly *after* the top-level model and never retrains or touches it.

**Deliberately-not-built, documented instead of coded** (don't
"fix" these without re-reading why first): URL unshortening
(`preprocessing/url_tools.py`) is fully built and tested but *not* called
from the live `/classify` path — synchronous redirect-following would
make request latency attacker-controlled (see `ClassificationPipeline`'s
docstring). The SBERT+vector-DB secondary outbreak layer and periodic
HDBSCAN batch clustering mentioned in the original design are out of
scope, per `docs/outbreak.md`. Full network-level SSRF isolation (vs. the
current IP-range blocklist) and most of KVKK/production operational
concerns are written up in `docs/production_readiness.md` rather than
implemented — that document is Turkish (its content is meant for
non-technical/legal review) while the rest of the docs are English.

**Do not use `BaranKanat/BerTurk-SpamSMS`** (or any of its weights) as a
base checkpoint anywhere — it's CreativeML OpenRAIL-M licensed, which
forbids the commercial use this project is scoped for. Details in
`docs/licensing_notes.md`.

**Small-dataset whack-a-mole is a known, recurring dynamic, not a bug to
silently "fix" with one seed edit**: with only ~80 hand-written synthetic
seeds underpinning the fraud-subtype labels, nudging one confusable
message pattern reliably perturbs another. `docs/model.md` documents a
full real investigation chain of this (including that broadening
`legitimate` example *diversity* — not fine-grained-class *volume* —
turned out to be what actually resolved a persistent case). Read it
before assuming a single new seed example is a permanent fix; validate
against the regression-style checks the doc describes.
