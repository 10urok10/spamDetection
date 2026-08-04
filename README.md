# spamdet

Turkish spam / fraud / gambling-scam / phishing detection MVP. Multi-stage
project; this README covers **Stage 1 (data + preprocessing)**,
**Stage 2 (model training + outbreak detection)**, and **Stage 3
(FastAPI gateway + review dashboard + Docker)**.

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
  merge.py                combines loader outputs into one deduplicated table
  build_dataset.py         scripts/build_dataset.py entry point
  generate_synthetic.py    scripts/generate_synthetic.py entry point
scripts/                  thin CLI wrappers + inspect_raw.py debug helper
data/
  raw/         gitignored - manually downloaded source files
  processed/   gitignored - merge.py / model.dataset output
  synthetic/
    seeds/       committed - hand-authored YAML seed examples
    generated/   gitignored - augment/adversarial script output
  review/      gitignored - confirmed.jsonl (human-approved review items)
models/        gitignored - train_model.py / export_onnx.py output
Dockerfile            api service image (CPU-only torch/onnxruntime)
docker-compose.yml    redis + api services
docs/
  datasets.md              per-source download/column/license notes
  licensing_notes.md       BerTurk-SpamSMS OpenRAIL-M warning - do not use it
  model.md                 Stage 2 model choice, GPU setup note, metrics caveat
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
- **Synthetic seed text is ASCII-only Turkish** (no `ç/ğ/ı/ö/ş/ü`
  diacritics) - deliberately, since a large share of real Turkish
  SMS/spam traffic is typed without them; diacritic handling itself is
  covered separately by the homoglyph/zero-width test suites.
- Rule-based paraphrasing (`TemplateParaphraser`) is intentionally simple
  and offline; an LLM-backed `Paraphraser` can be swapped in later without
  changing `augment_examples()`'s signature.
