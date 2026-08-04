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

As of Stage 2, no Kaggle datasets have been downloaded (see
`docs/datasets.md`), so `build_training_dataframe()` combines only:

- `turkishsms_ds` (HuggingFace, ~850 rows, live network call)
- Synthetic data: 61 hand-authored seeds -> paraphrase augmentation ->
  homoglyph/zero-width adversarial variants (~1500 rows after dedup)

A 4-epoch training run on this data reached **test f1_macro ~0.96** - but
this number is optimistic and should not be read as production-grade
accuracy: the train/val/test split (`model.dataset.split_dataset`) is
stratified by label *after* augmentation, so paraphrased/adversarial
variants of the *same* seed sentence can land in both train and test.
The model may partly be memorizing seed-sentence structure rather than
generalizing. This will improve once the Kaggle datasets are added (more
real-world examples) and/or the split is changed to group by original seed
before splitting - noted here rather than fixed now, since the Stage 2 goal
was a working pipeline, not a tuned model.

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
