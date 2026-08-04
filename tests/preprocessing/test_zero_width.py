import random

from spamdet.preprocessing.zero_width import (
    ZERO_WIDTH_CHARS,
    clean_zero_width,
    contains_zero_width,
    inject_zero_width,
    strip_zero_width,
)

ZWSP = chr(0x200B)
ZWNJ = chr(0x200C)
ZWJ = chr(0x200D)


def test_strip_removes_single_zero_width_char():
    assert strip_zero_width("b" + ZWSP + "ahis") == "bahis"


def test_strip_removes_mix_of_zero_width_chars_splitting_a_word():
    corrupted = ZWSP.join(["b", "a", ZWNJ, "h", "i", ZWJ, "s"])
    assert strip_zero_width(corrupted) == "bahis"


def test_strip_is_noop_when_none_present():
    text = "Merhaba, nasılsın? Bu normal bir mesajdır."
    assert strip_zero_width(text) == text


def test_strip_does_not_touch_normal_whitespace_or_turkish_letters():
    text = "Çok teşekkürler, iyi günler dilerim ığüşöç"
    assert strip_zero_width(text) == text


def test_contains_zero_width():
    assert contains_zero_width("bonus" + ZWSP) is True
    assert contains_zero_width("bonus") is False


def test_clean_zero_width_report_counts_and_lists_removed_chars():
    corrupted = "be" + ZWSP + "dava" + ZWJ + " bonus"
    report = clean_zero_width(corrupted)
    assert report.cleaned == "bedava bonus"
    assert report.removed_count == 2
    assert set(report.removed_chars) == {ZWSP, ZWJ}
    assert report.original == corrupted


def test_inject_zero_width_round_trips_with_strip():
    original = "Tebrikler kazandiniz hemen tiklayin"
    rng = random.Random(42)
    corrupted = inject_zero_width(original, fraction=0.5, rng=rng)
    assert corrupted != original
    assert strip_zero_width(corrupted) == original


def test_inject_zero_width_deterministic_given_seed():
    original = "bonus kazandiniz"
    a = inject_zero_width(original, fraction=0.5, rng=random.Random(7))
    b = inject_zero_width(original, fraction=0.5, rng=random.Random(7))
    assert a == b


def test_inject_zero_width_only_uses_known_chars():
    original = "bahis sitesi bonusu"
    corrupted = inject_zero_width(original, fraction=0.8, rng=random.Random(1))
    inserted = set(corrupted) - set(original)
    assert inserted <= ZERO_WIDTH_CHARS
