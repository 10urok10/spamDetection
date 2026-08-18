from spamdet.preprocessing.input_markers import mark_all
from spamdet.preprocessing.mersis_marker import MERSIS_MARKER
from spamdet.preprocessing.shortener_marker import SHORTENER_MARKER


def test_applies_both_markers_when_both_present():
    text = "Odeme icin: https://bit.ly/487moOS Mersis:0265017991000011"
    marked = mark_all(text)
    assert MERSIS_MARKER in marked
    assert SHORTENER_MARKER in marked
    assert text in marked


def test_applies_only_mersis_marker_when_only_mersis_present():
    text = "Kacirma! Firsat seni bekliyor. Mersis:0265017991000011"
    marked = mark_all(text)
    assert MERSIS_MARKER in marked
    assert SHORTENER_MARKER not in marked


def test_applies_only_shortener_marker_when_only_shortener_present():
    text = "Odeme icin: https://bit.ly/487moOS"
    marked = mark_all(text)
    assert SHORTENER_MARKER in marked
    assert MERSIS_MARKER not in marked


def test_noop_when_neither_present():
    text = "Yarin saat 15:00'te toplantimiz var, unutma."
    assert mark_all(text) == text
