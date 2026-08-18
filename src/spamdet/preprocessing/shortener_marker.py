from urllib.parse import urlsplit

from .url_tools import extract_urls

# Well-known generic/public URL-shortening services - domains anyone can
# register a link on, giving zero sender-identity signal. Deliberately
# does NOT include a company's own branded shortlink domain (e.g.
# hpj.im/Hepsijet, app.hb.biz/Hepsiburada, mgrs.link/Migros, qnb.mn/QNB -
# all seen in this project's real reklam/bilgilendirme seed data) - those
# already carry a real identifiable-sender signal the model can learn
# directly from the surrounding brand name, and lumping them in here
# would blur exactly the distinction this marker exists to draw: "a
# link anyone could have posted" vs. "a link this specific company
# controls." See docs/model.md for the phishing example (a fake-cargo-
# COD-payment message using bit.ly) that motivated this.
KNOWN_SHORTENER_DOMAINS = frozenset(
    {
        "bit.ly",
        "tinyurl.com",
        "cutt.ly",
        "dub.sh",
        "goo.gl",
        "is.gd",
        "ow.ly",
        "t.co",
        "rebrand.ly",
        "shorturl.at",
        "tiny.cc",
        "buff.ly",
        "bit.do",
        "soo.gd",
        "s.id",
        "adf.ly",
        "tr.im",
        "v.gd",
        "clck.ru",
        "qr.ae",
        "shorte.st",
        "rb.gy",
        "cli.re",
        "shrtco.de",
    }
)

# Prepended to the model's input text (not to stored/displayed
# cleaned_text - see model/train.py and model/inference.py), same
# soft-signal pattern as preprocessing/mersis_marker.py: an explicit
# "this message uses a generic, sender-anonymous link shortener" hint
# rather than a hard rule. Deliberately not a spam/phishing verdict on
# its own - real reklam/bilgilendirme messages have used generic
# shorteners too (see the real gambling-spam and reklam seed examples
# that both cite cutt.ly/dub.sh) - the model is still free to weigh it
# against everything else in the message.
SHORTENER_MARKER = "[SHORTENER_VAR]"


def _domain_of(url: str) -> str:
    candidate = url if "://" in url else f"http://{url}"
    host = urlsplit(candidate).hostname or ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def has_generic_shortener(text: str) -> bool:
    return any(_domain_of(url) in KNOWN_SHORTENER_DOMAINS for url in extract_urls(text))


def mark_shortener(text: str) -> str:
    if has_generic_shortener(text):
        return f"{SHORTENER_MARKER} {text}"
    return text
