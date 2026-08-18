"""Composes all soft tokenizer-input markers (mersis_marker,
shortener_marker, ...) into one function, applied identically at train
time (model/train.py) and serve time (model/inference.py) so they stay
in sync - see either marker module's docstring for why these are soft
signals prepended to the tokenizer's input, not hard classification
rules. Adding a new marker later only needs a change here, not at both
call sites.
"""

from .mersis_marker import mark_mersis
from .shortener_marker import mark_shortener


def mark_all(text: str) -> str:
    return mark_shortener(mark_mersis(text))
