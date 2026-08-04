import unicodedata

from spamdet.preprocessing.homoglyphs import (
    detect_mixed_script,
    is_homoglyph_attack,
    normalize_nfkd,
    strip_confusables,
)

CYRILLIC_A = chr(0x0430)  # CYRILLIC SMALL LETTER A, looks identical to Latin 'a'
CYRILLIC_E = chr(0x0435)  # CYRILLIC SMALL LETTER IE, looks identical to Latin 'e'


def test_nfkd_decomposes_turkish_i_with_dot_without_data_loss():
    # Turkish capital dotted I (U+0130) decomposes into base I + combining dot above.
    decomposed = normalize_nfkd(chr(0x0130))
    assert decomposed == "I" + chr(0x0307)
    # recomposing (NFC) must round-trip back to the original letter
    assert unicodedata.normalize("NFC", decomposed) == chr(0x0130)


def test_pure_turkish_text_is_not_flagged_as_mixed_script():
    text = "Hesabinizin sifresini degistirmeniz gerekiyor. Cok tesekkurler igüşöç"
    report = detect_mixed_script(text)
    assert report.is_mixed_script is False
    assert report.flagged_chars == []


def test_turkish_text_with_native_diacritics_is_not_flagged():
    text = "Çok teşekkürler, iyi günler dilerim ığüşöç"
    assert is_homoglyph_attack(text) is False


def test_cyrillic_substituted_into_latin_word_is_flagged():
    spoofed = "p" + CYRILLIC_A + "ypal hesabınız"
    report = detect_mixed_script(spoofed)
    assert report.is_mixed_script is True
    assert len(report.flagged_chars) >= 1
    assert report.flagged_chars[0]["character"] == CYRILLIC_A


def test_pure_single_script_cyrillic_text_is_not_flagged():
    # No Latin characters present -> not "mixed" script, even though every
    # character is non-Latin. We detect *mixing*, not "any non-Latin".
    pure_cyrillic = CYRILLIC_A + CYRILLIC_E + CYRILLIC_A
    assert is_homoglyph_attack(pure_cyrillic) is False


def test_strip_confusables_fixes_only_the_substituted_char():
    spoofed = "hes" + CYRILLIC_A + "b" + CYRILLIC_E + " şok kampanya"
    cleaned = strip_confusables(spoofed)
    assert CYRILLIC_A not in cleaned
    assert CYRILLIC_E not in cleaned
    assert "şok" in cleaned  # Turkish diacritics must survive untouched


def test_strip_confusables_is_noop_on_clean_turkish_text():
    text = "hesabınızda şüpheli işlem tespit edildi"
    assert strip_confusables(text) == text
