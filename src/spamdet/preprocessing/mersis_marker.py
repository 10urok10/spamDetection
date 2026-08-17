import re

# Mersis (Merkezi Sicil Kayit Sistemi) numbers are the Turkish trade
# registry id that commercial electronic messages (ticari elektronik
# ileti - KVKK-regulated ads) are legally required to display, usually
# 16 digits after a "Mersis:"/"Mersis No:"/"MRS:"/"MN:" label. Real
# examples seen in this project's data use all four label spellings
# (Hepsiburada/QNB/Getir use "Mersis:", Migros uses "MN:", a local LPG
# distributor uses "MRS:") - the regex covers all of them rather than
# just the literal word "mersis".
_MERSIS_PATTERN = re.compile(r"\b(?:mersis(?:\s*no)?|mrs|mn)\s*:\s*\d{10,16}\b", re.IGNORECASE)

# Prepended to the model's input text (not to stored/displayed
# cleaned_text - see model/train.py and model/inference.py) so the
# classifier gets an explicit, unambiguous "a Mersis number is present"
# signal instead of having to infer significance from a bare digit
# sequence competing with much stronger lexical signals elsewhere in the
# message (e.g. "TIKLA KAZAN"-style openers that read as spam/gambling
# on their own). This is a soft, experimental feature-injection
# technique, not a hard classification rule - Mersis presence alone
# does NOT reliably imply reklam (a real Flixbus customer-satisfaction
# survey has a Mersis number and is bilgilendirme, not reklam - see
# otp_rule.py's docstring note), so the model is still free to weigh it
# against everything else in the message rather than short-circuiting on
# it. Validate the usual regression suites after retraining with this
# enabled - it changes what the model sees, so it needs the same
# whack-a-mole scrutiny as any other data/preprocessing change.
MERSIS_MARKER = "[MERSIS_VAR]"


def has_mersis_number(text: str) -> bool:
    return bool(_MERSIS_PATTERN.search(text))


def mark_mersis(text: str) -> str:
    if has_mersis_number(text):
        return f"{MERSIS_MARKER} {text}"
    return text
