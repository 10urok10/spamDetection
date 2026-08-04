# Licensing notes

## Do not use BerTurk-SpamSMS

`BaranKanat/BerTurk-SpamSMS` on Hugging Face is a pretrained Turkish
SMS-spam classifier trained on the same Turkish SMS Collection dataset we
use in Stage 1. It is licensed under **CreativeML OpenRAIL-M**, which
prohibits commercial use/resale of the model and its derivative weights.

**Do not use this model, load its weights, or use it (or any derivative of
it) as a starting checkpoint anywhere in this project.** Stage 2 must
fine-tune from a base model with a license compatible with the project's
intended use (e.g. mDeBERTa-v3-base or a BERTurk *base* checkpoint - verify
license terms for whichever base model is chosen before training), not from
BerTurk-SpamSMS.

## Dataset license/attribution

Each public dataset in `docs/datasets.md` has its own license/attribution
terms (mostly academic/research-oriented Kaggle and Hugging Face uploads).
Verify and record the actual license for each dataset at download time -
this file currently only documents the one confirmed hard constraint
(BerTurk-SpamSMS) flagged in the project brief; the dataset-level terms
still need to be checked and filled in before any wider distribution of
this project or its trained models.
