from spamdet.preprocessing.mersis_marker import MERSIS_MARKER, has_mersis_number, mark_mersis


def test_detects_mersis_colon_format():
    assert has_mersis_number("Kacirma! Firsat seni bekliyor. Mersis:0265017991000011") is True


def test_detects_mrs_abbreviation():
    assert has_mersis_number("SADECE 1600 TL. MRS:0080001324800017") is True


def test_detects_mn_abbreviation():
    assert has_mersis_number("Kampanya detaylari icin tikla. MN:0622052951300016") is True


def test_detects_mersis_no_variant():
    assert has_mersis_number("Detayli bilgi: magaza.com. Mersis No: 0987654321000011") is True


def test_case_insensitive():
    assert has_mersis_number("mersis:0265017991000011") is True


def test_plain_text_without_mersis_is_not_flagged():
    assert has_mersis_number("Yarin saat 15:00'te toplantimiz var, unutma.") is False


def test_bare_digit_sequence_without_label_is_not_flagged():
    # a 16-digit number alone (e.g. a card/account number) must not
    # false-trigger just because it happens to be the right length
    assert has_mersis_number("Kart numaraniz: 0265017991000011") is False


def test_mark_mersis_prepends_marker_when_present():
    text = "Kacirma! Firsat seni bekliyor. Mersis:0265017991000011"
    marked = mark_mersis(text)
    assert marked == f"{MERSIS_MARKER} {text}"


def test_mark_mersis_is_noop_when_absent():
    text = "Yarin saat 15:00'te toplantimiz var, unutma."
    assert mark_mersis(text) == text
