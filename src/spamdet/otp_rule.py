import re

# A 4-8 digit standalone code (OTP codes are almost always in this range;
# longer sequences are typically account/contract/Mersis numbers, shorter
# ones are usually just small quantities like "%20" or "1250 TL").
_OTP_CODE_PATTERN = re.compile(r"\b\d{4,8}\b")

_OTP_KEYWORD_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"dogrulama\s*kodu",
        r"doğrulama\s*kodu",  # dogrulama with the Turkish g-breve
        r"onay\s*kodu",
        r"giriş\s*şifre",
        r"giris\s*sifre",
        r"tek\s*kullan[iı]ml[iı]k\s*şifre",
        r"tek\s*kullanimlik\s*sifre",
        r"güvenlik\s*kodu",
        r"guvenlik\s*kodu",
        r"verification\s*code",
        r"\botp\b",
        r"access\s*code",
        r"security\s*code",
        r"kimseyle\s*paylaş",
        r"kimseyle\s*paylas",
        r"başkas[iı]yla\s*paylaş",
        r"baskasiyla\s*paylas",
    ]
]

# NOTE: Mersis-number and opt-out ("RET yaz") detection were deliberately
# NOT added here as a rule for the reklam category. Real data disproved
# the assumption they're ad-specific: a real Flixbus customer-satisfaction
# survey message (see data/synthetic/seeds/bilgilendirme.yaml) contains
# BOTH a Mersis number and a "RET yaz" opt-out phrase, despite not being
# an advertisement - these turn out to be general regulated-bulk-SMS
# compliance markers, not ad-specific ones. They're still useful
# *signals*, just not deterministic ones - the reklam/bilgilendirme/spam
# distinction is left entirely to the ML classifier, which can weigh them
# alongside promotional vocabulary instead of triggering on them alone.


def detect_otp(text: str) -> bool:
    """True if ``text`` looks like a one-time-password/verification-code
    message: a 4-8 digit code together with a disclaimer/labeling phrase.
    Requiring both avoids false-triggering on messages that merely
    contain some 4-8 digit number (an amount, a contract number, ...)
    without actually being an OTP. This runs BEFORE the ML model - OTP
    is deterministic/templated enough that a rule is more reliable (and
    cheaper) than training a class for it.
    """
    if not _OTP_CODE_PATTERN.search(text):
        return False
    return any(pattern.search(text) for pattern in _OTP_KEYWORD_PATTERNS)
