# Legitimate-message subtype classification (otp / bilgilendirme / reklam)

## Why this exists

Built for an SMS-operator compliance use case: telling apart OTP,
informational (bilgilendirme), and advertisement (reklam) traffic among
messages the main classifier (`spamdet.model`) already calls
`legitimate`, so advertisement content can be checked against the right
(separately KVKK-consented) sending channel. This is explicitly **not**
about judging whether an ad was sent with proper consent - only whether a
message *is* an advertisement at all, confirmed by the user as the
scope.

`spam`/`gambling_scam`/`phishing`/`financial_urgency` never get a
subtype - only `legitimate` messages do.

## Architecture

```
[Asama A: spamdet.model - unchanged, never retrained]
        │
        ├── legitimate ──► [Kural: OTP - subtype.rules.detect_otp]
        │                      4-8 digit code + disclaimer phrase
        │                      -> otp, done
        │                          │ (not OTP)
        │                          ▼
        │                  [ML: AdInfoClassifier]
        │                      TF-IDF + logistic regression
        │                      -> reklam | bilgilendirme
        │
        ├── spam ──► same subtype check runs, but only OVERRIDES the
        │            "spam" verdict into "legitimate" if the result is
        │            confident enough (see "The spam-override problem"
        │            below) - otherwise stays spam, no subtype
        │
        └── gambling_scam / phishing / financial_urgency ──► never
            reconsidered, no subtype check at all (safety boundary)
```

The top-level fraud/spam classifier (`spamdet.model`) is **not**
retrained by any of this - the subtype layer only ever reads its output.

## The spam-override problem (a real gap found via live testing)

First version of this feature only ran the subtype layer when Stage A
said `legitimate`. Live-testing immediately showed the flaw: Stage A
(never retrained) still calls real, obvious ads like the Hepsiburada
coupon `spam` - so they never reached the subtype layer at all, and the
whole point of the feature (finding ads regardless of what bucket Stage
A put them in) silently failed for exactly the messages that motivated
it.

Fix: the subtype layer also runs when Stage A says `spam` (never for the
fraud-specific labels - those are left alone entirely). If the result is
confident enough, it **overrides** Stage A's `spam` into `legitimate`
with that subtype attached; a `PredictionResult` is reconstructed with
`label="legitimate"` and `confidence` set to Stage A's own already-computed
P(legitimate) (not a fabricated number), while `probabilities` is left
as Stage A produced it, so the original spam-leaning distribution stays
visible for audit. See `_should_override_spam_verdict` in
`api/pipeline.py`.

`otp` matches always override (deterministic rule). `reklam` only
overrides above `SPAM_TO_REKLAM_OVERRIDE_THRESHOLD`
(`api/config.py`) - calibrated against real data, not guessed: a first
attempt at 0.75 missed most genuine ads (real P(reklam) scores range
~0.51-0.81; the flagship real Hepsiburada example itself scored only
0.64), while bilgilendirme examples measured ~0.27-0.37. Landed on
**0.5** - clears the whole observed reklam range with margin below it,
still higher than the routine within-`legitimate` `reklam_threshold`
(0.4), since overriding a `spam` call is a bigger claim than refining a
`legitimate` one. `bilgilendirme` results never override - too weak a
signal to move a `spam` verdict on its own.

## A real finding that shaped this design: Mersis number is not ad-specific

The original plan (see project history) was a third rule: "Mersis
number and/or an opt-out phrase (RET yaz) -> reklam", on the theory that
these are ad-compliance markers. A real message the user supplied
disproved it: a genuine Flixbus customer-satisfaction survey invite
contains **both** a Mersis number and a "RET yaz" opt-out phrase, and is
not an advertisement (see the `bilgilendirme`-tagged entry with this note
in `data/synthetic/seeds/legitimate.yaml`). These turn out to be general
regulated-bulk-SMS compliance markers, not exclusively ad markers.

Consequence: **no Mersis/opt-out rule was built.** Those signals are left
as ordinary TF-IDF features for the ML classifier to weigh alongside
actual promotional vocabulary (indirim, kampanya, hediye, fırsat, kupon,
...), rather than a hard-coded trigger. Verified this works: both the
full real Flixbus message and a full real Vodafone KVKK disclosure
message (both containing Mersis numbers, neither an ad) classify
correctly as `bilgilendirme` with low P(reklam) (0.22-0.29) - see
`tests/subtype/`.

## Why TF-IDF + logistic regression, not a transformer fine-tune

Per the project's own "evaluate the cheap thing first" approach (see
`docs/model.md`): `AdInfoClassifier` is a plain scikit-learn
`Pipeline(TfidfVectorizer, LogisticRegression)`, not another mDeBERTa
fine-tune. Cheaper to train (seconds, no GPU), cheaper to serve (no
onnxruntime/torch needed just for this), and - checked, not assumed -
it generalizes well on held-out data (see below). If real-world usage
later shows it underperforming, escalating to a dedicated fine-tuned
model is the documented fallback, not a redesign.

## Evaluation

**Leak-free split**: like the main classifier's original training-data
caveat (`docs/model.md`), paraphrase-augmented variants of the same seed
sentence must not cross the train/test boundary. `subtype/train.py`
splits at the **raw seed level first**, then augments each split
independently - not the "augment first, split after" pattern that
inflated the main classifier's first-round numbers.

**Held-out test split** (47 raw seeds -> ~185 train / ~50 test after
augmentation), `reklam_threshold=0.4`:

| | precision | recall |
|---|---|---|
| reklam | 0.800 | **1.000** |
| bilgilendirme | 1.000 | 0.833 |

**Genuinely novel sentences never seen in any seed or augmented form**,
plus the two full real Mersis-containing messages above: **10/10
correct**, including both hard negatives.

The precision/recall trade-off is a real, deliberate choice, not an
accident: `reklam_threshold=0.4` (< 0.5) trades some `bilgilendirme`
precision for guaranteed `reklam` recall, matching the user's explicit
priority ("kaçırmamak önemli" - missing a real ad matters more than
double-checking a borderline informational message). Raise
`--reklam-threshold` back toward 0.5+ if false-positive volume on
`bilgilendirme` turns out to be a bigger operational cost than expected
in practice.

## Running it

```
python -m spamdet.subtype.train    # -> models/subtype-ad-info.joblib + .metrics.json
```

Reads the `subtype:` tags in `data/synthetic/seeds/legitimate.yaml`
(otp entries are excluded - rule-detected, not part of this classifier's
training data). The API (`api/app.py`) loads this file automatically at
startup if present; if absent, `/classify` simply omits the `subtype`
field (graceful degradation, same pattern as `build_dataset.py`'s
missing-source skips) rather than failing to start.

`models/subtype-ad-info.joblib` is gitignored, like the other trained
models - re-run the command above to reproduce it.
