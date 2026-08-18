from spamdet.preprocessing.shortener_marker import (
    SHORTENER_MARKER,
    has_generic_shortener,
    mark_shortener,
)


def test_detects_bitly():
    assert has_generic_shortener("Odeme icin: https://bit.ly/487moOS") is True


def test_detects_bare_domain_without_scheme():
    assert has_generic_shortener("Detay: cutt.ly/mngkargo9") is True


def test_detects_with_www_prefix():
    assert has_generic_shortener("Tikla: https://www.tinyurl.com/xyz123") is True


def test_case_insensitive():
    assert has_generic_shortener("https://BIT.LY/487moOS") is True


def test_does_not_flag_branded_company_shortlink():
    # hpj.im (Hepsijet's own domain) is not a generic public shortener -
    # it already carries a real-brand identity signal directly.
    assert has_generic_shortener("Takip icin: hpj.im/gm0fzrtj") is False


def test_does_not_flag_plain_text_without_urls():
    assert has_generic_shortener("Yarin saat 15:00'te toplantimiz var, unutma.") is False


def test_mark_shortener_prepends_marker_when_present():
    text = "Odeme icin: https://bit.ly/487moOS"
    assert mark_shortener(text) == f"{SHORTENER_MARKER} {text}"


def test_mark_shortener_is_noop_when_absent():
    text = "Takip icin: hpj.im/gm0fzrtj"
    assert mark_shortener(text) == text
