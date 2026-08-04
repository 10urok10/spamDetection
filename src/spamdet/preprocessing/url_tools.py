import ipaddress
import re
import socket
import urllib.parse
from dataclasses import dataclass, field
from typing import Literal

import requests

_SCHEME_URL_RE = re.compile(r'(?i)\bhttps?://[^\s<>"\']+')
_BARE_DOMAIN_RE = re.compile(
    r'(?i)\b(?:[a-z0-9-]+\.)+'
    r'(?:com|net|org|tr|com\.tr|gov\.tr|info|xyz|top|click|link|ly|gl|me|io|biz|co)\b'
    r'(?:/[^\s<>"\']*)?'
)
_TRAILING_PUNCT_RE = re.compile(r'[.,;:!?)\]\'"]+$')


def extract_urls(text: str) -> list[str]:
    """Extract http(s) URLs and bare shortener-style domains from text."""
    urls: list[str] = []
    for match in _SCHEME_URL_RE.finditer(text):
        urls.append(_TRAILING_PUNCT_RE.sub("", match.group(0)))
    for match in _BARE_DOMAIN_RE.finditer(text):
        candidate = _TRAILING_PUNCT_RE.sub("", match.group(0))
        if not any(candidate in u for u in urls):
            urls.append(candidate)
    return urls


class SSRFError(Exception):
    """Raised when a URL/redirect hop resolves to a disallowed scheme, host,
    or IP address (private/loopback/link-local/reserved/multicast/CGNAT)."""


ALLOWED_SCHEMES = {"http", "https"}
DEFAULT_TIMEOUT: tuple[float, float] = (3.05, 5.0)  # (connect, read) seconds
DEFAULT_MAX_REDIRECTS = 5
_CGNAT_RANGE = ipaddress.ip_network("100.64.0.0/10")


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    # Unwrap IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1) - a common
    # SSRF-check bypass if left unhandled.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if isinstance(ip, ipaddress.IPv4Address) and ip in _CGNAT_RANGE:
        return True
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _check_url_safe(url: str) -> urllib.parse.SplitResult:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ALLOWED_SCHEMES:
        raise SSRFError(f"disallowed scheme {parts.scheme!r}")
    if "@" in (parts.netloc or ""):
        raise SSRFError("userinfo in URL is not allowed")
    hostname = parts.hostname
    if not hostname:
        raise SSRFError("could not parse hostname from URL")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        addr_infos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise SSRFError(f"DNS resolution failed for {hostname!r}: {exc}") from exc
    for _family, _type, _proto, _canonname, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if _is_blocked_ip(ip):
            raise SSRFError(f"{hostname!r} resolves to disallowed address {sockaddr[0]}")
    return parts


@dataclass(frozen=True)
class UnshortenResult:
    original_url: str
    final_url: str
    redirect_chain: list[str] = field(default_factory=list)
    status: Literal["ok", "blocked_ssrf", "too_many_redirects", "timeout", "error"] = "ok"
    error: str | None = None


def unshorten(
    url: str,
    *,
    session: requests.Session | None = None,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
    proxies: dict[str, str] | None = None,
) -> UnshortenResult:
    """Follow redirects to resolve a (possibly shortened) URL, refusing to
    connect to any hop that resolves to a private/internal/link-local
    address (SSRF guard).

    Known limitation (documented, not solved here): there is a small
    TOCTOU window between our DNS check and `requests`' own connection -
    full protection against DNS-rebinding requires routing this call
    through an isolated egress proxy/network (planned for Stage 3
    Dockerization). The `proxies` parameter exists so that can be added
    later without changing this function's call sites.

    Response bodies are never read - only headers are needed to follow
    redirects, which also limits attack surface.
    """
    session = session or requests.Session()
    chain: list[str] = []
    current = url
    headers = {"User-Agent": "spamdet-stage1/0.1"}
    try:
        for _ in range(max_redirects + 1):
            _check_url_safe(current)
            chain.append(current)
            resp = session.head(current, allow_redirects=False, timeout=timeout, proxies=proxies, headers=headers)
            if resp.status_code == 405:
                resp = session.get(
                    current, allow_redirects=False, timeout=timeout, proxies=proxies, headers=headers, stream=True
                )
                resp.close()
            if resp.status_code in (301, 302, 303, 307, 308) and "Location" in resp.headers:
                current = urllib.parse.urljoin(current, resp.headers["Location"])
                continue
            return UnshortenResult(url, current, chain, "ok")
        return UnshortenResult(url, current, chain, "too_many_redirects")
    except SSRFError as exc:
        return UnshortenResult(url, current, chain, "blocked_ssrf", str(exc))
    except requests.Timeout as exc:
        return UnshortenResult(url, current, chain, "timeout", str(exc))
    except requests.RequestException as exc:
        return UnshortenResult(url, current, chain, "error", str(exc))
