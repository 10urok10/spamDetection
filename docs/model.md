# Model (Stage 2)

## Choice: mDeBERTa-v3-base

Fine-tuned `microsoft/mdeberta-v3-base` (multilingual, disentangled
attention) rather than a Turkish-only BERTurk checkpoint - the training
data includes English reference examples alongside Turkish, and
mDeBERTa-v3's architecture is a strong general-purpose default. Either
choice is reasonable; this was a Stage-2 assumption made without blocking,
per project convention.

**Do not use `BaranKanat/BerTurk-SpamSMS`** as a starting checkpoint or
weights source - see `docs/licensing_notes.md`.

## Windows/Blackwell GPU note

`pip install torch` alone resolves to a CPU-only wheel, or a CUDA build too
old for Blackwell GPUs (RTX 50-series, compute capability sm_120 - PyTorch
reports "does not include kernels for this GPU"). The working install on
this machine was:

```
pip install torch==2.13.0+cu132 --index-url https://download.pytorch.org/whl/cu132
```

Check `nvidia-smi` for your driver's supported CUDA version and match it to
the closest `cuXXX` index at https://download.pytorch.org/whl/. Also
requires `sentencepiece` (not pulled in automatically by `transformers` in
all versions) for the DeBERTa-v2 tokenizer.

## Focal Loss

`src/spamdet/model/focal_loss.py` implements multi-class focal loss (Lin et
al. 2017) plus inverse-frequency class weights, used because
`gambling_scam`/`phishing`/`financial_urgency` have far fewer examples than
`legitimate`/`spam`. `FocalLossTrainer` is a thin `transformers.Trainer`
subclass overriding `compute_loss`.

## Training data caveat - read before trusting the reported metrics

**Update (2026-08, Stage 3 live testing)**: the 3 Kaggle datasets have
since been downloaded (see `docs/datasets.md` for the real-format
surprises this exposed - semicolon delimiters, a malformed CSV needing an
.xlsx fallback, etc.) and `build_training_dataframe()` now combines:

- `turkishsms_ds` (HuggingFace, ~850 rows)
- `turkish_sms_collection`, `turkish_spam_dataset`, `sms_spam_collection`
  (Kaggle, ~10,647 rows combined)
- Synthetic data: 74 hand-authored seeds -> paraphrase augmentation ->
  homoglyph/zero-width adversarial variants (~600 rows after dedup)

A 4-epoch run on this larger, mostly-real dataset (9,850 train / 1,232
val / 1,232 test) reached **test f1_macro ~0.989** with much better-
calibrated confidence scores on manual spot checks (correct predictions
now typically 0.84-0.97, versus 0.68-0.9 on the old mostly-synthetic
model) - a genuine improvement, not just a number. The rest of this
section describes the *original* (pre-Kaggle-data) caveat, kept for
context on why train/test leakage was a real concern before:

The original (61-seed, ~1500-row, synthetic-heavy) 4-epoch training run
reached **test f1_macro ~0.96** - but
this number is optimistic and should not be read as production-grade
accuracy: the train/val/test split (`model.dataset.split_dataset`) is
stratified by label *after* augmentation, so paraphrased/adversarial
variants of the *same* seed sentence can land in both train and test.
The model may partly be memorizing seed-sentence structure rather than
generalizing. This will improve once the Kaggle datasets are added (more
real-world examples) and/or the split is changed to group by original seed
before splitting - noted here rather than fixed now, since the Stage 2 goal
was a working pipeline, not a tuned model.

## Observed overconfidence (Stage 3 finding)

Live-testing the Stage 3 API against real requests, the model turned out
to be confidently "legitimate" (0.7-0.87) even on near-empty or nonsense
input (`"Merhaba"`, `"tamam"`, `"1000 TL"`) - consistent with the
train/test leakage caveat above (small, synthetic-heavy training data ->
a model that's very sure of itself rather than well-calibrated). In
practice this means the review queue's 0.4-0.6 confidence band rarely
triggers with the current model; it's a real signal that recalibration
(e.g. temperature scaling) or more diverse training data should be a
priority before relying on the confidence score for anything
safety-critical.

## Whack-a-mole failure mode from live user testing (Stage 3 finding)

Two rounds of real user-reported misclassifications exposed a direct
consequence of the small seed set:

1. A URL-less phishing threat ("Şimdi doğrulama yapmazsanız hesabınız 24
   saat içinde kapatılacaktır!") was misclassified `legitimate` at 0.87 -
   traced to **all 14** original phishing seeds containing a URL, so the
   model had learned "phishing implies a link" rather than the threat/
   urgency language itself. Fix: added URL-less phishing/vishing seeds.
2. That fix's side effect: two genuinely legitimate messages (a real
   utility e-invoice notification and a real app OTP/quick-login code)
   started getting misclassified `spam` (~0.47-0.48) - the model's
   decision boundary shifted toward flagging formal/security-toned
   language in general, because `legitimate` didn't have enough matching
   diversity to counterbalance the new phishing examples. Fix: added
   matching legitimate examples (bill notification, OTP with the
   "GUVENLIGINIZ ICIN KIMSEYLE PAYLASMAYINIZ" disclaimer - itself a
   standard *legitimate* security phrase, not a phishing tell).
3. After fix 2, the original URL-less phishing example from (1) drifted
   back toward `legitimate` (0.40 vs 0.397 - essentially a tie).

**Takeaway**: with ~74 hand-written seeds, nudging the decision boundary
for one confusable pair reliably perturbs another - this is expected
behavior for a dataset this size, not a bug to keep patching example by
example. Treat any single hand-added seed example as a spot-fix with a
blast radius, not a guaranteed permanent fix, until the dataset is
meaningfully larger.

**Update after adding the Kaggle datasets**: retraining on the full
~10,600-real-row dataset fixed the two legitimate-message false positives
outright (now 0.69-0.84 confidence, comfortably correct) and sharply
improved calibration everywhere else - but the *same* URL-less phishing
example is still misclassified `legitimate`, now even more confidently
(0.79). Root cause is structural, not a seed-wording problem: `spam`/
`legitimate` now have ~10,600 real examples behind them, while
`gambling_scam`/`phishing`/`financial_urgency` still have only the ~74
synthetic seeds - so growing the real-data classes without proportionally
growing the synthetic fine-grained ones made the model *more* certain
about the coarse categories at the fine-grained categories' expense. This
won't be fixed by more seed examples alone; it needs either (a) a real
Turkish phishing/fraud dataset with fine-grained labels (doesn't appear to
exist publicly - see `docs/datasets.md`), or (b) a much larger synthetic
generation pass (hundreds, not dozens, of seeds per fine-grained
category) to rebalance relative volume against the now much bigger
binary-labeled classes.

## Running it

```
python scripts/train_model.py               # fine-tune, ~1-2 min on a modern GPU for this data size
python scripts/export_onnx.py                # export models/spamdet-mdeberta -> models/spamdet-mdeberta-onnx
```

`--offline` skips the `turkishsms_ds` network call (synthetic data only).
Both scripts accept `--raw-dir`/`--seed-dir` overrides matching
`build_dataset.py`'s conventions. `SpamClassifier`
(`src/spamdet/model/inference.py`) loads either a PyTorch checkpoint or an
ONNX export directory (auto-detected by the presence of `model.onnx`) and
was verified to produce identical predictions on both backends, with ONNX
roughly 3x faster on this machine for small batches.

`models/` is gitignored - re-run the scripts above to reproduce it.
