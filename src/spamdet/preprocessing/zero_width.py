import random
import re
from dataclasses import dataclass

# Invisible / zero-width characters commonly used to split spam trigger
# words and evade keyword-based filters (e.g. "b<ZWSP>ahis" -> "bahis").
# Built from explicit codepoints (rather than embedding the literal
# invisible characters in this file) so the source stays unambiguous
# in editors/diffs regardless of encoding.
ZERO_WIDTH_CODEPOINTS = (
    0x200B,  # ZERO WIDTH SPACE
    0x200C,  # ZERO WIDTH NON-JOINER
    0x200D,  # ZERO WIDTH JOINER
    0x2060,  # WORD JOINER
    0xFEFF,  # ZERO WIDTH NO-BREAK SPACE / BOM
    0x180E,  # MONGOLIAN VOWEL SEPARATOR
    0x200E,  # LEFT-TO-RIGHT MARK
    0x200F,  # RIGHT-TO-LEFT MARK
    0x2061,  # FUNCTION APPLICATION
    0x2062,  # INVISIBLE TIMES
    0x2063,  # INVISIBLE SEPARATOR
    0x2064,  # INVISIBLE PLUS
)
ZERO_WIDTH_CHARS = frozenset(chr(cp) for cp in ZERO_WIDTH_CODEPOINTS)

_ZW_PATTERN = re.compile("[" + "".join(ZERO_WIDTH_CHARS) + "]")


@dataclass(frozen=True)
class ZeroWidthReport:
    original: str
    cleaned: str
    removed_count: int
    removed_chars: list[str]


def clean_zero_width(text: str) -> ZeroWidthReport:
    removed = _ZW_PATTERN.findall(text)
    return ZeroWidthReport(
        original=text,
        cleaned=_ZW_PATTERN.sub("", text),
        removed_count=len(removed),
        removed_chars=removed,
    )


def strip_zero_width(text: str) -> str:
    return _ZW_PATTERN.sub("", text)


def contains_zero_width(text: str) -> bool:
    return _ZW_PATTERN.search(text) is not None


def inject_zero_width(text: str, *, fraction: float = 0.2, rng: random.Random | None = None) -> str:
    """Insert zero-width characters between letters, for generating
    adversarial training/test variants (the evasion this module defends
    against). ``fraction`` is the approximate share of internal gaps
    (between two non-space characters) that get a zero-width char inserted.
    """
    rng = rng or random.Random()
    chars = list(ZERO_WIDTH_CHARS)
    out = []
    for i, ch in enumerate(text):
        out.append(ch)
        if i == len(text) - 1:
            continue
        nxt = text[i + 1]
        if ch.isspace() or nxt.isspace():
            continue
        if rng.random() < fraction:
            out.append(rng.choice(chars))
    return "".join(out)
