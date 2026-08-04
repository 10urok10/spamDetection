import unicodedata
from dataclasses import dataclass, field

from confusable_homoglyphs import confusables
from confusable_homoglyphs.categories import alias as script_alias

# Turkish letters (c-cedilla, g-breve, dotless-i, o/u-diaeresis, dotted-I)
# decompose under NFKD into base-letter + combining mark. NFKD here is used
# ONLY to canonicalize representation (so precomposed and decomposed forms
# compare equal); it must never be followed by stripping combining marks or
# re-encoding to ASCII, or real Turkish words would be silently corrupted
# (e.g. "sok" and the real word "sok" would become indistinguishable from
# a corrupted "sok" that used to be "sok" via decomposition of "sok").
# Also: never .lower()/.casefold() Turkish text here - Python's default
# casing does not know the Turkish I/i vs I-dotless/i-dotted distinction.


def normalize_nfkd(text: str) -> str:
    return unicodedata.normalize("NFKD", text)


@dataclass(frozen=True)
class HomoglyphReport:
    text: str
    is_mixed_script: bool
    flagged_chars: list[dict] = field(default_factory=list)


def detect_mixed_script(text: str, *, preferred_aliases: tuple[str, ...] = ("LATIN",)) -> HomoglyphReport:
    """Flags text that mixes scripts in a way that's suspicious, i.e.
    contains non-Latin characters that could be confused with Latin ones
    (the classic Cyrillic-lookalike phishing/spam evasion pattern).

    Pure Turkish text (Latin + Turkish-specific letters) is never flagged:
    confusable_homoglyphs categorizes Turkish letters as LATIN, so they
    don't register as a second "mixed" script.
    """
    normalized = normalize_nfkd(text)
    # is_dangerous() only returns a bool; is_confusable(greedy=True) is what
    # gives us the actual per-character detail, so we call it directly and
    # gate on is_mixed_script() ourselves (this is exactly what is_dangerous
    # does internally, just without discarding the detail).
    confusable_chars = confusables.is_confusable(
        normalized, greedy=True, preferred_aliases=list(preferred_aliases)
    )
    flagged_list = list(confusable_chars) if confusable_chars else []
    is_mixed = bool(flagged_list) and confusables.is_mixed_script(normalized)
    return HomoglyphReport(text=text, is_mixed_script=is_mixed, flagged_chars=flagged_list if is_mixed else [])


def is_homoglyph_attack(text: str, *, preferred_aliases: tuple[str, ...] = ("LATIN",)) -> bool:
    return detect_mixed_script(text, preferred_aliases=preferred_aliases).is_mixed_script


def strip_confusables(text: str, *, preferred_aliases: tuple[str, ...] = ("LATIN",)) -> str:
    """Replace only characters flagged as cross-script confusables with a
    same-script (preferred_aliases) look-alike, leaving every other
    character - including Turkish diacritics - untouched.
    """
    report = detect_mixed_script(text, preferred_aliases=preferred_aliases)
    if not report.is_mixed_script:
        return text

    replacements: dict[str, str] = {}
    preferred_upper = {a.upper() for a in preferred_aliases}
    for entry in report.flagged_chars:
        char = entry["character"]
        for candidate in entry.get("homoglyphs", []):
            candidate_char = candidate.get("c")
            if not candidate_char or len(candidate_char) != 1:
                continue
            if script_alias(candidate_char) in preferred_upper:
                replacements[char] = candidate_char
                break

    normalized = normalize_nfkd(text)
    substituted = "".join(replacements.get(ch, ch) for ch in normalized)
    # Recompose back to NFC: substitutions above ran against the NFKD form
    # (base letter + combining marks), so untouched Turkish letters that got
    # decomposed (e.g. "s" + combining-cedilla) need to be recomposed back
    # into their precomposed form ("s-cedilla") to match ordinary text.
    return unicodedata.normalize("NFC", substituted)
