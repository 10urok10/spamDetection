import random

from spamdet.preprocessing.homoglyphs import detect_mixed_script
from spamdet.preprocessing.zero_width import ZERO_WIDTH_CHARS, clean_zero_width
from spamdet.schema import Label, Lang, Record
from spamdet.synthetic.adversarial import (
    generate_adversarial_set,
    make_homoglyph_variant,
    make_zero_width_variant,
)


def test_homoglyph_variant_differs_and_is_flagged_by_detector():
    original = "paypal hesabinizi dogrulayin"
    variant = make_homoglyph_variant(original, fraction=0.8, rng=random.Random(1))
    assert variant != original
    report = detect_mixed_script(variant)
    assert report.is_mixed_script is True


def test_homoglyph_variant_zero_fraction_is_noop():
    original = "test metni"
    variant = make_homoglyph_variant(original, fraction=0.0, rng=random.Random(1))
    assert variant == original


def test_zero_width_variant_round_trips_with_cleaner():
    original = "bahis sitesine hemen uye ol"
    variant = make_zero_width_variant(original, fraction=0.5, rng=random.Random(2))
    inserted = set(variant) - set(original)
    assert inserted <= ZERO_WIDTH_CHARS
    assert clean_zero_width(variant).cleaned == original


def test_generate_adversarial_set_reproducible_given_seed():
    seeds = [Record(text="bonus kazandiniz tiklayin", label=Label.GAMBLING_SCAM, source="synthetic_seed", lang=Lang.TR)]
    a = generate_adversarial_set(seeds, rng_seed=42)
    b = generate_adversarial_set(seeds, rng_seed=42)
    assert [r.text for r in a] == [r.text for r in b]


def test_generate_adversarial_set_emits_homoglyph_and_zero_width_pair_per_seed():
    seeds = [
        Record(text="sifrenizi guncelleyin", label=Label.PHISHING, source="synthetic_seed", lang=Lang.TR),
    ]
    records = generate_adversarial_set(seeds, rng_seed=1)
    assert len(records) == 2
    variant_types = {r.extra["variant_type"] for r in records}
    assert variant_types == {"homoglyph", "zero_width"}
    for r in records:
        assert r.source == "synthetic_adversarial"
        assert r.label is Label.PHISHING
        assert r.extra["original_text"] == "sifrenizi guncelleyin"
