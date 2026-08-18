# spamdet

Turkish SMS/message classification MVP. Flat 4-category taxonomy - `otp`,
`reklam`, `spam`, `bilgilendirme` (`src/spamdet/schema.py`'s `Label`) - no
fraud-subtype or legitimate-subtype hierarchy. Multi-stage project; this
README covers **Stage 1 (data + preprocessing)**, **Stage 2 (model
training + outbreak detection)**, and **Stage 3 (FastAPI gateway + review
dashboard + Docker)**.

## Stage 1 scope

- Ingest 5 public spam/ham datasets (3 Turkish, 2 English reference) into a
  single normalized schema (`text, label, source, lang`) - `ham` maps to
  `bilgilendirme`, `spam` maps to `spam` (see `docs/datasets.md`)
- Hand-authored + augmented + adversarial synthetic data for the two
  categories no public Turkish dataset distinguishes (`otp`, `reklam`),
  including deliberately spam-*looking* `bilgilendirme` examples (real
  bank/cargo notifications) to guard against false positives
- Pre-processing modules: homoglyph/mixed-script normalizer, zero-width
  character cleaner, URL extractor + SSRF-safe unshortener

## Stage 2 scope

- mDeBERTa-v3-base fine-tuning with focal loss for class imbalance
  (`src/spamdet/model/`) - see `docs/model.md` for the GPU/CUDA setup note
  and an important caveat about the reported metrics
- ONNX export + a backend-agnostic inference wrapper
  (`SpamClassifier`, verified to give identical predictions on both
  PyTorch and ONNX backends)
- SimHash + Redis LSH-banding outbreak (near-duplicate blast) detection
  (`src/spamdet/outbreak/`) - see `docs/outbreak.md` for what's
  implemented vs. explicitly deferred (SBERT secondary layer, periodic
  HDBSCAN batch clustering)

## Stage 3 scope

- FastAPI gateway (`src/spamdet/api/`) wiring Stage 1 preprocessing ->
  Stage 2 classifier -> Stage 2 outbreak detector into one `/classify`
  endpoint, with confidence-band (0.4-0.6) routing to a Redis-backed
  human review queue
- Server-rendered review dashboard (`GET /dashboard`, no JS) - list
  pending items, approve (optionally with a corrected label, which is
  appended to `data/review/confirmed.jsonl` for future retraining) or
  reject
- Prometheus metrics at `GET /metrics` (message volume by label,
  confidence distribution, review-queue and outbreak-alert counters)
- `Dockerfile` + `docker-compose.yml` (`redis` + `api` services)
- `docs/production_readiness.md` ("Uretime Gecis Notu", in Turkish):
  KVKK/data retention, rough infra cost estimate, model drift/retraining
  strategy, human-in-the-loop operations - documented, not implemented,
  per the project's explicit MVP scope

## Label taxonomy

Four flat, mutually-exclusive categories, no hierarchy:

- **`otp`** - one-time password / login code messages. Detected by a
  deterministic rule (`src/spamdet/otp_rule.py`: digit code + disclaimer
  phrase), never by the ML model - it's templated/structured enough that
  a rule beats a learned class and needs no training data.
- **`reklam`** - advertisements/marketing, regardless of KVKK-consent
  status - this system only judges whether a message *is* an ad, not
  whether it was sent through the right consented channel.
- **`spam`** - unsolicited/unwanted messaging that isn't an ad, including
  what used to be broken out as `gambling_scam`/`phishing`/
  `financial_urgency` (now folded into this one label).
- **`bilgilendirme`** - everything else legitimate/informational (bank
  notifications, cargo tracking, appointment reminders, ...).

`otp`/`reklam`/`spam`/`bilgilendirme` are predicted by a single
`ClassificationPipeline` (`src/spamdet/api/pipeline.py`): the OTP rule
runs first and short-circuits the ML model when it matches; otherwise the
3-way mDeBERTa fine-tune (`bilgilendirme`/`reklam`/`spam`) runs. See
`docs/model.md` for why this replaced an earlier 5-label +
legitimate-subtype design, and for a real finding on why no
Mersis-number/opt-out-phrase rule exists for `reklam` (a real customer-
satisfaction-survey message disproved that shortcut).

## Setup

Requires a dedicated conda environment (Python 3.11):

```
conda create -n spamdet python=3.11
conda activate spamdet
pip install -e ".[dev]"                     # Stage 1 only
pip install -e ".[dev,train,outbreak,api]"  # everything
```

Stage 2 also needs a CUDA-matched `torch` build (see `docs/model.md`) and
`sentencepiece`. Stage 2/3 need a running Redis (`docker compose up -d
redis`) for anything beyond each module's own (fully offline,
`fakeredis`-based) tests.

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

## Training the model

```
python scripts/train_model.py    # fine-tune -> models/spamdet-mdeberta/
python scripts/export_onnx.py    # export -> models/spamdet-mdeberta-onnx/
```

`--offline` skips the `turkishsms_ds` network call. See `docs/model.md`.

## Outbreak (near-duplicate blast) detection

```
docker compose up -d redis
```

```python
import redis
from spamdet.outbreak.detector import OutbreakDetector

detector = OutbreakDetector(redis.Redis(host="localhost", port=6379))
result = detector.ingest("msg-id-1", "Tebrikler! Bonus kazandiniz...")
# result.is_outbreak_candidate, result.similar_message_ids, result.similarities
```

See `docs/outbreak.md` for the LSH band-width tuning rationale and what's
deliberately not implemented yet.

## Manually relabeling public-dataset rows

The public spam/ham datasets are binary, so the loaders blanket-map every
row to `Label.SPAM`/`Label.BILGILENDIRME` (see `docs/datasets.md`). A
manual check found the "spam"-labeled bucket is overwhelmingly real
advertising, not fraud - the datasets' original annotators used "spam" to
mean any bulk/commercial SMS. `scripts/label_tool.py` is a standalone
local tool (no model/Redis dependency) for relabeling candidates by hand:

```
python scripts/label_tool.py    # http://localhost:8010/label
```

One message at a time, keyboard shortcuts 1/2/3/4/0 (bilgilendirme/otp/
reklam/spam/skip). Progress saves incrementally to
`data/manual_labels/relabeled.jsonl` (gitignored - real message text,
same rationale as `data/review/`), safe to stop and resume. For bulk-
classifying the remainder of the spam bucket via a narrow, hand-verified
keyword list (gambling/adult-scam terms only - **not** a broad "predatory
loan" rule, which false-positived on real bank card campaigns):

```
python scripts/bulk_label_spam_bucket.py            # dry run, prints counts
python scripts/bulk_label_spam_bucket.py --commit    # actually writes decisions
```

`model/dataset.py`'s `build_training_dataframe()` picks up
`data/manual_labels/relabeled.jsonl` automatically (merged with priority
over the public loaders' default mapping) - no flag needed, missing file
is a silent no-op. See `manual_labels.py`'s module docstring and
`docs/model.md`'s 2026-08 update for the full finding. The "ham" bucket
has not been reviewed yet - a real remaining opportunity, lower priority
since it's already mostly genuine bilgilendirme.

## Running the API

Local (no Docker):

```
docker compose up -d redis      # dependency
python scripts/run_api.py       # http://localhost:8000
```

Full stack (API + Redis) via Docker - requires `models/` to already exist
locally (train first, see above; the compose file mounts it read-only):

```
docker compose up -d --build
```

Key endpoints: `POST /classify`, `GET /review/pending`, `POST
/review/{item_id}/decide`, `GET /dashboard`, `GET /metrics`, `GET /health`.

```
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Tebrikler! Bonus kazandiniz, hemen tiklayin: bit.ly/x"}'
```

**`GET /demo`** - a small interactive page (type a message, see the
predicted category, per-category probability bars, and the
homoglyph/review-queue/outbreak flags update live via `fetch`). Separate
from `/dashboard` on purpose: `/dashboard` is the no-JS human-review tool,
`/demo` is a JS-based classify-anything sandbox for trying the model out.

## Project layout

```
src/spamdet/
  schema.py              Label/Lang/Record - the normalized record schema
  otp_rule.py             OTP rule (digit code + disclaimer phrase), no training data
  loaders/                one module per public dataset + shared base.py
  preprocessing/
    homoglyphs.py          NFKD canonicalization + mixed-script detection
    zero_width.py          invisible-character stripping (+ injection for
                            adversarial generation)
    url_tools.py            URL extraction + SSRF-safe redirect unshortening
    mersis_marker.py         soft "Mersis number present" tokenizer-input signal
    shortener_marker.py       soft "generic link-shortener present" signal
    input_markers.py          mark_all() - composes the two markers above,
                               used identically by model/train.py and inference.py
  manual_labels.py        candidate pool + JSONL round-trip for the relabeling tool
  synthetic/
    seeds.py                loads data/synthetic/seeds/*.yaml
    augment.py               rule-based paraphrase augmentation
    adversarial.py            homoglyph/zero-width adversarial variant generation
  model/
    labels.py                shared Label <-> id mapping (training/inference)
    dataset.py                combines loaders + synthetic data, stratified split
    focal_loss.py              FocalLoss + FocalLossTrainer (transformers.Trainer)
    train.py                   fine-tuning entry point
    export_onnx.py               ONNX export via optimum
    inference.py                  SpamClassifier (PyTorch or ONNX, auto-detected)
  outbreak/
    simhash.py                64-bit SimHash fingerprinting
    lsh.py                     RedisLSHIndex - band-based candidate lookup
    detector.py                 OutbreakDetector - ingest + near-duplicate check
  api/
    config.py                  env-var config (model dir, Redis URL, ...)
    schemas.py                  Pydantic request/response models
    pipeline.py                  ClassificationPipeline - the per-message flow
    review_queue.py               Redis-backed human review queue
    metrics.py                     Prometheus counters/histograms
    app.py                          create_app() factory + all routes
    templates/dashboard.html         server-rendered review dashboard
    templates/demo.html               interactive JS classify-anything sandbox
    templates/label.html              scripts/label_tool.py's relabeling UI
  merge.py                combines loader outputs into one deduplicated table
  build_dataset.py         scripts/build_dataset.py entry point
  generate_synthetic.py    scripts/generate_synthetic.py entry point
scripts/                  thin CLI wrappers + inspect_raw.py debug helper;
                           label_tool.py + bulk_label_spam_bucket.py (see above)
data/
  raw/         gitignored - manually downloaded source files
  processed/   gitignored - merge.py / model.dataset output
  synthetic/
    seeds/       committed - hand-authored YAML seed examples (otp/reklam/spam/bilgilendirme)
    generated/   gitignored - augment/adversarial script output
  manual_labels/  gitignored - relabeled.jsonl (scripts/label_tool.py output)
  review/      gitignored - confirmed.jsonl (human-approved review items)
models/        gitignored - train_model.py / export_onnx.py output
Dockerfile            api service image (CPU-only torch/onnxruntime)
docker-compose.yml    redis + api services
docs/
  datasets.md              per-source download/column/license notes
  licensing_notes.md       BerTurk-SpamSMS OpenRAIL-M warning - do not use it
  model.md                 Stage 2 model choice, GPU setup note, metrics caveat,
                            and the 2026-08 flat-taxonomy pivot
  outbreak.md               Stage 2 outbreak layer design + what's deferred
  production_readiness.md    Stage 3 "Uretime Gecis Notu" (Turkish)
```

## Known limitations (by design, documented not solved)

- **`url_tools.unshorten()` is not wired into `/classify`**: it's built,
  tested (Stage 1), and SSRF-safe, but resolving redirect chains over the
  network can take seconds - calling it synchronously in the classify
  request path would make API latency depend on an attacker-controlled
  server. `ClassificationPipeline` only extracts raw URLs; see its
  docstring and `docs/production_readiness.md` for the deferred
  async-enrichment alternative.
- **SSRF guard DNS-rebinding TOCTOU**: `url_tools.unshorten()` validates
  each redirect hop's resolved IP before connecting, but there's a small
  window between that check and `requests`' own connection. Full
  protection requires routing outbound requests through an isolated
  egress proxy/network - not built (see `docs/production_readiness.md`);
  the `proxies` parameter on `unshorten()` exists so that can be wired in
  without changing call sites.
- **Kaggle downloads are manual**: no `kaggle` API credential handling in
  code by default (the `datafetch` extra installs the `kaggle` CLI
  separately if you want to script it yourself). Raw files are just
  expected to already exist under `data/raw/`.
- **Synthetic seed text is mostly ASCII-only Turkish** (no `ç/ğ/ı/ö/ş/ü`
  diacritics) - deliberately, since a large share of real Turkish
  SMS/spam traffic is typed without them; diacritic handling itself is
  covered separately by the homoglyph/zero-width test suites. Formal-
  register content (KVKK notices, account-status updates) is an exception
  and is written with real diacritics on purpose - that register is
  reliably written that way in practice, and testing it via an ASCII-
  transliterated version was found to mask real classification gaps
  (see `docs/model.md`'s 2026-08-18 update).
- Rule-based paraphrasing (`TemplateParaphraser`) is intentionally simple
  and offline; an LLM-backed `Paraphraser` can be swapped in later without
  changing `augment_examples()`'s signature.
- **No automated regression-test gate on the trained model.** All live
  verification so far has been manual/conversational (POST real messages
  to a running `/classify`, eyeball the label) - there's no checked-in
  fixed test set of confirmed (label, text) pairs a future retrain gets
  automatically checked against. Combined with real train-to-train
  variance (identical data + a pinned seed does not guarantee identical
  model behavior - GPU op non-determinism), a retrain's pass/fail on any
  informal check should not be over-trusted in isolation. Building that
  checked-in regression set is the most valuable next step before more
  ad-hoc fixing - see `CLAUDE.md`'s "Status" section for the fuller list
  of currently-known-fragile patterns.
