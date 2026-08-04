import random

from ..preprocessing.zero_width import inject_zero_width
from ..schema import Record

SOURCE_NAME = "synthetic_adversarial"

# Curated Latin -> confusable-lookalike map for GENERATION (deterministic,
# one-to-one). This is intentionally distinct from confusable_homoglyphs'
# much larger many-to-many DETECTION database used in preprocessing.homoglyphs
# - here we only need a handful of visually-identical Cyrillic substitutes
# for the Latin letters most commonly swapped in real evasion attempts.
LATIN_TO_CONFUSABLE: dict[str, str] = {
    "a": chr(0x0430),  # CYRILLIC SMALL LETTER A
    "e": chr(0x0435),  # CYRILLIC SMALL LETTER IE
    "o": chr(0x043E),  # CYRILLIC SMALL LETTER O
    "p": chr(0x0440),  # CYRILLIC SMALL LETTER ER
    "c": chr(0x0441),  # CYRILLIC SMALL LETTER ES
    "x": chr(0x0445),  # CYRILLIC SMALL LETTER HA
    "y": chr(0x0443),  # CYRILLIC SMALL LETTER U
    "i": chr(0x0456),  # CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
}


def make_homoglyph_variant(text: str, *, fraction: float = 0.3, rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    out = []
    for ch in text:
        substitute = LATIN_TO_CONFUSABLE.get(ch.lower())
        if substitute is not None and rng.random() < fraction:
            out.append(substitute)
        else:
            out.append(ch)
    return "".join(out)


def make_zero_width_variant(text: str, *, fraction: float = 0.2, rng: random.Random | None = None) -> str:
    return inject_zero_width(text, fraction=fraction, rng=rng)


def generate_adversarial_set(
    examples: list[Record], *, rng_seed: int = 42, source: str = SOURCE_NAME
) -> list[Record]:
    """For each seed record, emit a homoglyph-corrupted and a
    zero-width-corrupted variant, tagged with enough ``extra`` metadata to
    trace back to the original text. Reproducible given ``rng_seed``.
    """
    rng = random.Random(rng_seed)
    out: list[Record] = []
    for rec in examples:
        homoglyph_text = make_homoglyph_variant(rec.text, rng=rng)
        out.append(
            Record(
                text=homoglyph_text,
                label=rec.label,
                source=source,
                lang=rec.lang,
                extra={**rec.extra, "variant_type": "homoglyph", "original_text": rec.text},
            )
        )
        zero_width_text = make_zero_width_variant(rec.text, rng=rng)
        out.append(
            Record(
                text=zero_width_text,
                label=rec.label,
                source=source,
                lang=rec.lang,
                extra={**rec.extra, "variant_type": "zero_width", "original_text": rec.text},
            )
        )
    return out
