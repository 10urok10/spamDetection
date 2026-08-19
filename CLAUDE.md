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
python scripts/label_tool.py                     # manual relabeling UI at :8010/label (see below)
python scripts/bulk_label_spam_bucket.py [--commit]  # bulk-classify the rest of the spam bucket

docker compose up -d redis      # Redis only, for local dev against scripts/run_api.py
docker compose up -d --build    # full api+redis stack (needs models/ present, mounted read-only)
```

No Docker was used this session (see the "Docker/WSL2" note above) - local dev/testing ran the API
directly via a scratchpad script calling `create_app(classifier=SpamClassifier(get_model_dir()),
redis_client=fakeredis.FakeStrictRedis(...))` on port 8000, bypassing Docker/real Redis entirely.
Recreate it if resuming: it's ~15 lines, see `api/config.get_model_dir` + `model.inference.SpamClassifier`
+ `fakeredis.FakeStrictRedis(decode_responses=False)` wired into `create_app()`. Kill stale listeners with
`Get-NetTCPConnection -LocalPort 8000 | Where State -eq Listen | % { Stop-Process -Id $_.OwningProcess -Force }`
before restarting after a retrain (the ONNX dir is loaded once at process startup, not hot-reloaded).

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
   generates the `otp`/`reklam` training data no public Turkish dataset
   has (the public datasets are binary spam/ham and can't distinguish
   these from generic `spam`/`bilgilendirme`): hand-written seeds in
   `data/synthetic/seeds/*.yaml` → rule-based paraphrase augmentation →
   homoglyph/zero-width adversarial corruption, all offline and
   deterministic given `--rng-seed`.

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

**Label taxonomy is a flat 4-way split, not a hierarchy** — `otp`,
`reklam`, `spam`, `bilgilendirme` (`schema.py`'s `Label`, `model/labels.py`
for the 3-way ML subset, fixed order = fixed integer ids everywhere). This
replaced an earlier, more elaborate design: a 5-label top level
(`legitimate, spam, gambling_scam, phishing, financial_urgency`) with a
separate post-hoc "legitimate-subtype" layer (`otp`/`bilgilendirme`/
`reklam` splitting whatever the top-level model called `legitimate`).
Per explicit user instruction ("4 kategori olacak, diğer şeyleri sil" —
there will be 4 categories, delete the other things), that whole hierarchy
was collapsed: the three fraud subtypes (`gambling_scam`/`phishing`/
`financial_urgency`) now just fall under generic `spam`, and the
subtype-detection layer (formerly `subtype/`) was deleted entirely and
folded into the main pipeline. Don't resurrect fraud-specific labels or a
subtype layer without a similarly explicit instruction to do so — this was
a deliberate simplification, not an oversight.

- **`otp`** — one-time password / login codes. Detected by a
  deterministic rule (`otp_rule.py`: digit code + disclaimer phrase,
  formerly `subtype/rules.py`), never predicted by the ML model — it's
  templated/structured enough that a rule beats a learned class and needs
  no training data. `model/dataset.py` filters `otp`-labeled seed rows out
  of ML training data for the same reason (see `model/labels.py`'s
  `LABELS`, which excludes it).
- **`reklam`** — advertisements, regardless of KVKK-consent status; this
  system only judges whether a message *is* an ad, not whether it was
  sent through the right consented channel. **No Mersis-number/opt-out-
  phrase rule exists for `reklam`** — a real user-supplied customer-
  satisfaction-survey message contains both markers without being an ad,
  disproving that shortcut; those signals are ML features (now trained
  directly into the main model), not a hard-coded trigger.
- **`spam`** — unsolicited/unwanted messaging that isn't an ad, including
  the former fraud subtypes (gambling scams, phishing, financial-urgency
  social engineering). `spam` has meant different things at different
  points in this project's history (see `docs/model.md`'s "superseded
  twice" note) — don't trust an old doc passage's characterization of
  `spam` without checking which era it's from.
- **`bilgilendirme`** — everything else legitimate/informational (bank
  notifications, cargo tracking, appointment reminders, ...). The public
  spam/ham datasets' `ham` maps here (see `docs/datasets.md`).

The ML model (`model/train.py`) is a 3-way classifier over
`bilgilendirme`/`reklam`/`spam` — `otp` is rule-only. `ClassificationPipeline`
(`api/pipeline.py`) runs the OTP rule first and short-circuits to the ML
model only when it doesn't match; there is no override/reclassification
logic layered on top of the model's own verdict anymore (the earlier
"spam-override-to-legitimate-subtype" mechanism was deleted along with the
subtype layer).

**Manual relabeling of public-dataset rows** (`manual_labels.py`,
`scripts/label_tool.py`, `scripts/bulk_label_spam_bucket.py`) — the single
highest-leverage fix found in this project's history so far. The public
Turkish datasets (`turkish_sms_collection`, `turkish_spam_dataset`) only
distinguish spam vs. ham, so the loaders blanket-map every row to
`Label.SPAM`/`Label.BILGILENDIRME`. A manual sample check of the
"spam"-labeled bucket (2,787 rows) found it's **overwhelmingly real
advertising** (KIGILI, Garanti Mortgage, Vatan, CarrefourSA, VakifBank
Worldcard, ...), not fraud - the original annotators evidently used
"spam" to mean any bulk/commercial SMS. That means the pipeline had been
training thousands of real ads as spam examples directly, a much bigger
structural cause of reklam-vs-spam confusion than any amount of synthetic
seed patching could fix. `scripts/label_tool.py` is a standalone local
FastAPI tool (`:8010/label`, no model/Redis dependency) for a human to
bulk-relabel candidates one at a time (keyboard shortcuts 1-4/0);
`scripts/bulk_label_spam_bucket.py` auto-classifies the rest via a narrow,
hand-verified keyword list (gambling/adult-scam terms - **not** a broad
"nakit avans"/"kart son 6 hane" rule, which had a bad false-positive rate
against real VakifBank/Paraf card campaigns that use the identical
mechanic) defaulting everything else to reklam. `model/dataset.py`'s
`build_training_dataframe()` merges `data/manual_labels/relabeled.jsonl`
in FIRST so a human-confirmed relabeling overrides the public loader's
default mapping for that exact text. The whole spam bucket (2,787 rows)
is now processed (178 manual + 2,522 bulk → reklam; 6 manual + 27 bulk →
spam; 53 → skip, garbled MIME email fragments); the **ham bucket (~2,702
rows) has never been reviewed** - a real remaining opportunity, lower
priority since ham is already mostly genuine bilgilendirme. The file is
gitignored (same rationale as `data/review/` - real message text
shouldn't be redistributed even though the source dataset is public), so
a fresh clone has none of this until someone re-runs the tool.

**Soft tokenizer-input markers, not hard rules** (`preprocessing/
mersis_marker.py`, `shortener_marker.py`, composed by `input_markers.py`'s
`mark_all()`, applied identically in `model/train.py` and
`model/inference.py` so train/serve never drift) - when regex detects a
Mersis-number pattern or a known generic-shortener domain (bit.ly,
cutt.ly, dub.sh, tinyurl.com, ... - deliberately excluding company-owned
branded shortlinks like `hpj.im`/`app.hb.biz`/`mgrs.link`, which already
carry a real identity signal), it prepends a marker token
(`[MERSIS_VAR]`/`[SHORTENER_VAR]`) to what the tokenizer sees. This is
explicitly **not** a classification rule - a real customer-satisfaction
survey has a Mersis number and is bilgilendirme, not reklam (see
`otp_rule.py`'s docstring), so the signal is just given to the model to
weigh, same as any other token. Same design as `otp_rule.py`'s hard rule
only where the pattern is genuinely deterministic (a code + disclaimer
phrase); everywhere else, prefer a soft signal or more real training data
over a hard trigger - a broad keyword rule for "predatory loan SMS" was
tried and rejected for exactly this reason (false-positived on real bank
cash-advance campaigns).

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
silently "fix" with one seed edit**: after the manual-relabeling work
above, `reklam` is no longer the smallest class (it's now the *second
largest*, ~3,900 training rows including the 2,700 real relabeled ones -
`otp` stays smallest but is rule-only so it doesn't matter), but the
dynamic itself is undiminished - it just moved to smaller sub-patterns
within a class (e.g. `financial_urgency.yaml`'s hijacked-relative scam
pattern, ~11 raw examples, got knocked out twice in one session by
unrelated `reklam` additions elsewhere). `docs/model.md` documents a full
real investigation chain of this from before the taxonomy pivot
(including that broadening a catch-all class's example *diversity* — not
a fine-grained class's *volume* — turned out to be what actually resolved
a persistent case; the lesson maps onto `bilgilendirme` today the same
way it mapped onto `legitimate` then, and now onto `financial_urgency`
within `spam`). Read it before assuming a single new seed example is a
permanent fix; validate against the regression-style checks the doc
describes - and see the next two notes, which qualify what "validate"
actually requires now.

**Training is not perfectly reproducible even with identical data and a
pinned seed** - confirmed directly in this project's history: reverting
`reklam.yaml` to a prior commit byte-for-byte and retraining did *not*
reproduce that commit's behavior; it landed *worse* (a real fraud message
that had been correctly `spam` was now `bilgilendirme` at 97%, plus a new
unrelated regression) than the state being reverted *from*. `TrainingArguments(seed=...)`
pins weight init and dataloader shuffling, but GPU op non-determinism
(cuDNN etc.) still causes real run-to-run variance on top of that. Practical
consequences: (1) "revert the data" is not the same operation as "revert the
model" - if a regression needs undoing, expect to retrain and re-verify, not
just `git checkout` the seed file and assume the old behavior comes back;
(2) a single retrain's pass/fail on the regression suites is not fully
trustworthy in isolation - a run that looks great might just be a lucky
draw, and a run that looks bad might not indict the data change that
preceded it; (3) there is currently no mechanism to select the best of
several training runs or otherwise control for this - a real gap, not
yet addressed.

**Test with real Turkish diacritics (ç ğ ı ö ş ü İ), not ASCII-transliterated
text** - this session's own regression-probe scripts used ASCII text
throughout (matching the synthetic-seed convention below) and it masked a
real, severe gap: formal/corporate-register bilgilendirme messages (KVKK
notices, account-status updates) scored dramatically differently with
diacritics restored than without (one flipped from correctly
`bilgilendirme` to wrongly `reklam` at 75%+ purely from that change).
This was *not* a general "diacritics break everything" problem (the
established regression suites passed identically either way) - just a
narrow, real, under-represented style that ASCII testing had been hiding.
`data/synthetic/seeds/*.yaml` is still deliberately ASCII-only per the
README's original rationale (informal real SMS often is), but formal-
register content added since (the Örüntü A/B `bilgilendirme.yaml` batch,
the KVKK-pattern `reklam.yaml` examples) is written with real diacritics
on purpose, and any live/manual verification of formal-register text
should use real diacritics, not a transliterated version.

## Status as of 2026-08-18 (read this before doing more live-testing/fixing)

The model at `models/spamdet-mdeberta(-onnx)/` right now is real, trained,
and - per manual live verification - passes every established regression
check with zero misses (16/16 known-regression cases, 10/10 novel-brand
reklam cases, 5/5 real user-supplied reklam messages, plus a 137-message
formal-register stress test the user generated). All of today's fixes are
committed and pushed to `master` (nothing local/uncommitted). But per the
training-variance note above, treat "currently passing" as *this specific
trained checkpoint's* state, not a permanent property of the seed data -
retraining from the current seeds is not guaranteed to reproduce it.

**Update: `scripts/check_regression.py` now exists** (committed same day as
this note) - 52 (label, text) pairs accumulated from this project's real
fix-and-retrain sessions, run by hand after any retrain
(`python scripts/check_regression.py`). Not part of `pytest` (loads the
model, too slow for the fast offline suite). This replaces the old
scratchpad-script-only verification approach; keep adding confirmed cases
to it going forward instead of leaving verification in disposable
scratchpad scripts again.

**Planned, not yet built: train multiple times and pick the best run.**
Confirmed the same day as this note (see the training-variance note
above): repeating `train_model.py` unmodified produces different
`check_regression.py` scores from run to run. The concrete plan, agreed
with the user but deliberately deferred rather than built immediately:
train N times to separate `--output-dir`s (optionally varying
`--rng-seed` per run rather than relying only on incidental GPU jitter),
run `check_regression.py` against each (point it at a given run via the
`SPAMDET_MODEL_DIR` env var to skip re-exporting ONNX for every
candidate), keep whichever run scores highest (tie-break: which specific
cases failed - a missed fraud/spam case is worse than a missed reklam-vs-
bilgilendirme boundary case - then the training script's own test
f1_macro), promote the winner to `models/spamdet-mdeberta`, export ONNX
only for that one, and delete the other runs' checkpoints (each is
sizeable). A real side-benefit beyond just picking a winner: a case that
fails in *every* run across repeated training is a genuine data gap
(needs more real examples), while a case that fails in only some runs is
just noise not worth chasing further - this is a fast, cheap way to tell
the two apart before spending more effort on any one "fragile" pattern.
Not yet turned into a script (e.g. `scripts/train_and_select.py`) - do
that before relying on this manually.

Known, not-yet-addressed gaps, roughly in priority order:
1. **No automated regression gate wired into anything.** Verification is
   manual (`check_regression.py` must be run by hand). Nothing stops a
   future retrain from silently regressing a previously-fixed case, and
   the multi-run-selection plan above isn't built yet.
2. **The "ham" bucket (~2,702 rows) has never been reviewed** via
   `scripts/label_tool.py` - real otp/reklam signal plausibly still hides
   there, mislabeled bilgilendirme, same as the spam bucket was.
3. **Several confirmed-fragile patterns remain genuinely unresolved**,
   not just "known" - each has been fixed and then regressed at least
   once: the "arkadaşını davet et" referral-bonus reklam pattern, bare/
   brandless short discount messages ("%40 indirim... KODU", no brand
   context), and the URL-less threat/urgency phishing pattern
   ("Şimdi doğrulama yapmazsanız..."). Don't assume any of these are
   solved just because the current checkpoint happens to pass on them.
4. **`financial_urgency.yaml`'s hijacked-relative scam pattern is thin**
   (~11 raw examples) and demonstrably sensitive to unrelated changes
   elsewhere - regressed twice in one session from `reklam.yaml` edits
   that had nothing to do with it.
5. **`otp_rule.py` doesn't distinguish OTP/login codes from other
   "doğrulama kodu"-labeled codes** - a courier delivery-confirmation
   code ("Teslimat için gerekli doğrulama kodu: 4072...") triggers the
   rule and gets labeled `otp`. Low real-world harm (still correctly
   "not spam") but semantically wrong; not fixed.
6. Beyond the specific formal-register gap already fixed, there's been
   no systematic sweep confirming every other category/pattern is robust
   to real Turkish diacritics vs. the ASCII form most seeds are written in.
