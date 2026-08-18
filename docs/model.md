# Model (Stage 2)

## Update (2026-08-18): soft feature markers, a formal-register bilgilendirme
## gap, and two methodology findings (training variance, diacritics testing)

Continuing from the manual-relabeling breakthrough below, four more things
came out of the same live-testing session (see `CLAUDE.md`'s "Status"
section for the full current-state handoff, which supersedes any specific
number below as more fixes land):

- **Soft tokenizer-input markers** (`preprocessing/mersis_marker.py`,
  `shortener_marker.py`, `input_markers.py`): a Mersis-number pattern or a
  known generic-link-shortener domain gets an explicit marker token
  prepended before tokenization, at both train and serve time. Same
  philosophy as `otp_rule.py`'s hard rule vs. everything else being a
  learned signal - these markers are hints, not verdicts, because both
  signals are genuinely ambiguous on their own (a real customer-survey
  message has a Mersis number and is bilgilendirme; real bank campaigns
  use the same "send card's last 6 digits" mechanic scam SMS use). A
  broader keyword rule for "predatory loan SMS" was tried and explicitly
  rejected mid-session for this reason - it false-positived on real
  VakifBank/Paraf card campaigns.
- **A formal-register bilgilendirme gap** ("Örüntü A": messages informing
  about an *existing* benefit/campaign's status - expiring, changing,
  non-renewing - vs. "Örüntü B": calm, no-urgency, no-link security/
  verification notices) - both share vocabulary with `reklam.yaml`/
  `phishing.yaml` almost verbatim but were being misclassified as those,
  sometimes even `spam`. Root cause was under-representation, not a
  vocabulary problem - fixed with ~20 real+synthetic `bilgilendirme.yaml`
  additions. The actual differentiator between this pattern and reklam
  turned out to be subtle and required real back-and-forth with the user
  to calibrate correctly (e.g. "your benefit expires today" read as
  bilgilendirme in isolation, but the user's explicit call was that
  urgently pushing someone to use it before it expires is itself
  promotional, i.e. reklam - kept as a deliberate contrast pair rather
  than assumed).
- **Training is not fully reproducible even with identical data and a
  pinned seed** - discovered by directly reverting a seed file to a prior
  commit and retraining, which reproduced a *worse* result than the state
  being reverted from (GPU op non-determinism on top of the pinned
  `TrainingArguments(seed=...)`). Reverting data is not the same operation
  as reverting model behavior; expect to retrain-and-reverify, and don't
  fully trust one run's pass/fail on the regression suites in isolation.
- **ASCII-transliterated testing was masking a real gap.** This session's
  own regression-probe scripts tested with ASCII text (matching the
  synthetic-seed convention), which is fine for casual-register content
  but hid how bad the formal-register bilgilendirme gap above actually
  was - one message scored correctly with ASCII and wrongly once real
  Turkish diacritics were restored. Confirmed this wasn't a general
  diacritics problem (all established regression suites held either way)
  - just evidence that ASCII testing isn't a safe stand-in for formal-
  register real-world text.

## Update (2026-08): the reklam class was fixed by real data, not more synthetic seeds

After the flat-taxonomy pivot below, `reklam` went through many rounds of
targeted synthetic-seed patching (adding examples for specific confusable
patterns: "TIKLA KAZAN"-style openers, "%40" vs "yuzde 40", formal KVKK
notices that mention marketing, referral-bonus ads, ...). Each round
measurably helped, but one pattern - referral-bonus ads structurally
resembling `gambling_scam.yaml`'s referral-bonus scams - survived several
rounds of dedicated fixing, including one attempt that was reverted after
it regressed three other previously-correct cases (a textbook whack-a-mole
instance, see `data/synthetic/seeds/reklam.yaml`'s own note on it).

The actual fix turned out not to be more synthetic seeds at all. A
throwaway comment from the user - "these are all bilgilendirme, there's
no reklam in here" while manually reviewing the public dataset's "ham"
bucket via `scripts/label_tool.py` - led to checking the "spam"-labeled
bucket instead (~2,787 real Turkish rows across `turkish_sms_collection`/
`turkish_spam_dataset`). It turned out to be **overwhelmingly real
advertising** from identifiable brands (KIGILI, Garanti Mortgage, Vatan,
CarrefourSA, VakifBank Worldcard, Paraf, ...), not fraud - only ~1-2% was
genuine gambling/scam content. These datasets' original annotators
evidently used "spam" to mean "any bulk/commercial SMS," not this
project's narrower definition. That meant **the training pipeline had
been feeding the model thousands of real ads labeled as spam**, directly
undermining every synthetic reklam seed added to counteract it - a much
bigger structural cause of the reklam-vs-spam confusion than anything
seed-level patching could fix, no matter how many rounds.

Fix: `manual_labels.py` + `scripts/label_tool.py`/`bulk_label_spam_bucket.py`
relabel that bucket (see their docstrings) and merge it into training
with priority over the public loaders' default mapping. After the user
hand-labeled a 185-row sample (96% reklam, confirming the finding) and a
narrow hand-verified keyword rule bulk-classified the rest, retraining on
~2,700 additional real reklam rows fixed *every* remaining known
confusable pattern in one pass, including the referral-bonus case that
had survived multiple dedicated synthetic-seed rounds - the first time
the project's 10-case novel-brand regression probe went 10/10. The
resulting test f1_macro (~0.969) is *lower* than earlier purely-synthetic
rounds (~0.99), and that's the honest, trustworthy number: the higher
ones were inflated by a test set that was mostly template-paraphrases of
~100 hand-written seeds (see "Training data caveat" below - this is the
same train/test-realism lesson recurring at a different layer). Moral,
consistent with this file's older findings: when real message diversity
is available, it beats synthetic seed patching outright rather than just
supplementing it - don't reach for another synthetic-seed round to fix a
persistent confusable pattern before checking whether relabeling
existing public data can do it instead.

## Update (2026-08): flat 4-category taxonomy

The label scheme changed after this doc's investigation history below was
written. Per explicit user decision, the earlier 5-label
(`legitimate`/`spam`/`gambling_scam`/`phishing`/`financial_urgency`) +
separate legitimate-subtype layer (`otp`/`bilgilendirme`/`reklam`, see the
now-deleted `docs/subtype.md`) design was **fully collapsed** into one flat
taxonomy: `otp`, `reklam`, `spam`, `bilgilendirme` (`schema.Label`). There
is no more "coarse vs. fine-grained" or "legitimate vs. its subtype"
distinction - these are four siblings.

- `otp` is still rule-detected (`spamdet.otp_rule.detect_otp`, formerly
  `subtype/rules.py`) and never reaches the ML model - see
  `model/labels.py`.
- The ML model is now a **3-way** classifier: `bilgilendirme`, `reklam`,
  `spam` (`model/labels.py`'s `LABELS`). `reklam` used to be a separate
  post-hoc subtype classifier (TF-IDF + logistic regression); it's now
  folded directly into the main mDeBERTa fine-tune as a first-class label.
- The three fraud subtypes (`gambling_scam`, `phishing`,
  `financial_urgency`) no longer exist as distinct labels - their seed
  files (`data/synthetic/seeds/gambling_scam.yaml` etc.) are retagged
  `category: spam` and still contribute spam-pattern diversity, just
  without a fine-grained label attached.

**The investigation history below (overconfidence, whack-a-mole,
train/test leakage, the Mersis-number finding) predates this pivot and
uses the old label names** (`legitimate`, `phishing`, `gambling_scam`,
etc.). It's kept because the *lessons* still apply directly to the new
taxonomy - e.g. "broadening a catch-all class's example diversity, not a
fine-grained class's volume, was the actual fix" maps onto
`bilgilendirme` today exactly as it mapped onto `legitimate` before, and
the Mersis-number-isn't-ad-specific finding is why `reklam` still has no
hard-coded Mersis/opt-out rule (see `spamdet/otp_rule.py`'s docstring and
`spamdet/subtype/ad_info_classifier.py`'s history, now folded into the
main model). Re-read this history before treating a single new seed
example as a permanent fix to a new-taxonomy confusion - the same
small-dataset dynamic is expected to resurface.

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
al. 2017) plus inverse-frequency class weights, used for class imbalance
across the 3-way ML label set (`model/labels.py`'s `LABELS`). Under the
current flat taxonomy `reklam` is by far the smallest/most imbalanced
class (~228 synthetic rows vs. thousands of `bilgilendirme`/`spam` rows
from the public datasets - see `docs/datasets.md`), previously it was the
three fraud subtypes vs. `legitimate`/`spam`. `FocalLossTrainer` is a thin
`transformers.Trainer` subclass overriding `compute_loss`, generic over
whatever `LABELS` currently contains - no code change needed when the
taxonomy changed.

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

**Final update - resolved via real user-collected examples**: rather than
(a)/(b) above, the actual fix ended up being targeted: the user supplied
9 real SMS messages from their own phone (cargo tracking, a telecom KVKK
disclosure, a subscription-cancellation confirmation, a benefit-card
onboarding message, a satisfaction-survey invite, a reward-credit
notification, an English OTP, and - deliberately kept as `spam` per the
user's judgment call - a Mersis-numbered, opt-out-compliant marketing
coupon). 5 of the 9 were misclassified `spam`. Adding all of them (minus
the one correctly-already-`spam` coupon) as `legitimate` seeds and
retraining fixed all 5 **and**, as a side effect, finally also fixed the
long-standing URL-less-phishing case from the previous round (now 0.956
confident and stable) - evidence that broadening `legitimate` example
*diversity* (not just volume) was the actual bottleneck, not fine-grained
class volume as hypothesized above. Full regression suite at this point:
**19/19** real+synthetic test cases correct. This does not mean the
whack-a-mole dynamic is solved in general - it means this specific round
of it responded well to real (not synthetic) counterexamples; expect the
same dynamic to resurface for new message styles not yet represented.

**Product decision made along the way (superseded twice since - see
below)**: legitimate, regulated marketing (Mersis number + opt-out
mechanism, e.g. a retailer coupon SMS) should classify as `spam`, not
`legitimate` - confirmed explicitly by the user at the time. This was
later reversed (ads moved back to `legitimate`, subtyped `reklam` - see
the now-deleted `docs/subtype.md`'s history, and the "hepsiburada kuponu"
example), and reversed again by the 2026-08 flat-taxonomy pivot at the
top of this doc: `reklam` is now its own top-level label, sibling to
`spam`/`otp`/`bilgilendirme`, not a spam variant or a legitimate subtype.
Fraud-specific intent (the old `gambling_scam`/`phishing`/
`financial_urgency`) is folded into generic `spam` today - `spam` no
longer has one stable definition across this doc's history, so don't
trust a `spam` characterization above without checking which era it's
from.

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
