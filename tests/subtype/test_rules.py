import pytest

from spamdet.subtype.rules import detect_otp


@pytest.mark.parametrize(
    "text",
    [
        "Tek kullanimlik sifreniz: 482910. Bu kodu kimseyle paylasmayin.",
        "GUVENLIGINIZ ICIN KIMSEYLE PAYLASMAYINIZ! Hizli Giris sifreniz 741852 ile uygulamaya giris yapabilirsiniz.",
        "Dogrulama kodunuz: 559214. Bu kodu kimseyle paylasmayiniz, sadece uygulama icinde kullaniniz.",
        "123456 is your Huawei verification code to register HUAWEI ID.",
        "Guvenlik kodunuz: 84213. Kimseyle paylasmayin.",
    ],
)
def test_detects_real_otp_patterns(text):
    assert detect_otp(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "22508007 sozlesme no'lu aboneliginizin odenmemis 1 adet elektrik fatura borcu 411.56 TL'dir.",
        "Apple iPhone icin 1250 TL kuponun var. Mersis:0265017991000011",
        "Hesabiniza 1.500,00 TL EFT girisi gerceklesmistir. Bakiye: 3.240,00 TL.",
        "Tebrikler! Bahis hesabiniza 500 TL bonus tanimlandi. Hemen cekmek icin: bit.ly/bonus500",
        "Yarin saat 15:00'te toplantimiz var, unutma.",
        "",
    ],
)
def test_does_not_false_trigger_on_non_otp_text_with_numbers(text):
    assert detect_otp(text) is False


def test_requires_both_code_and_keyword_not_either_alone():
    # keyword present, but no digit code at all
    assert detect_otp("Dogrulama kodunuz kimseyle paylasmayiniz.") is False
    # digit code present, but no OTP-signaling keyword
    assert detect_otp("Siparis numaraniz 482910 ile takip edebilirsiniz.") is False
