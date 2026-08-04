import socket
from unittest.mock import patch

import pytest
import responses

from spamdet.preprocessing.url_tools import (
    DEFAULT_MAX_REDIRECTS,
    _is_blocked_ip,
    extract_urls,
    unshorten,
)
import ipaddress


# --------------------------------------------------------------------------
# extract_urls
# --------------------------------------------------------------------------


def test_extract_single_url():
    text = "Hesabinizi dogrulayin: https://ornek-site.com/verify?id=123"
    assert extract_urls(text) == ["https://ornek-site.com/verify?id=123"]


def test_extract_multiple_urls():
    text = "Link 1: http://a.com/x Link 2: https://b.com/y"
    assert extract_urls(text) == ["http://a.com/x", "https://b.com/y"]


def test_extract_bare_shortener_domain():
    text = "Tebrikler! Odulunuzu almak icin tiklayin bit.ly/kazandiniz"
    assert extract_urls(text) == ["bit.ly/kazandiniz"]


def test_extract_strips_trailing_turkish_sentence_punctuation():
    text = "Bakiyenizi kontrol edin: https://ornek.com/bakiye."
    assert extract_urls(text) == ["https://ornek.com/bakiye"]


def test_extract_returns_empty_list_when_no_url():
    text = "Merhaba, yarin saat 15:00'te toplantimiz var."
    assert extract_urls(text) == []


def test_extract_does_not_duplicate_bare_domain_already_captured_by_scheme():
    text = "Tiklayin: https://bit.ly/abc123 hemen simdi"
    assert extract_urls(text) == ["https://bit.ly/abc123"]


# --------------------------------------------------------------------------
# SSRF IP blocklist
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ip_str",
    [
        "127.0.0.1",  # loopback
        "10.1.2.3",  # private
        "172.16.0.5",  # private
        "192.168.1.1",  # private
        "169.254.169.254",  # link-local / cloud metadata
        "100.64.0.1",  # CGNAT
        "0.0.0.0",  # unspecified
        "224.0.0.1",  # multicast
        "240.0.0.1",  # reserved
        "::1",  # ipv6 loopback
        "fe80::1",  # ipv6 link-local
        "fc00::1",  # ipv6 unique-local (private)
        "::ffff:127.0.0.1",  # ipv4-mapped ipv6 loopback bypass
    ],
)
def test_blocked_ips(ip_str):
    assert _is_blocked_ip(ipaddress.ip_address(ip_str)) is True


@pytest.mark.parametrize("ip_str", ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"])
def test_public_ips_not_blocked(ip_str):
    assert _is_blocked_ip(ipaddress.ip_address(ip_str)) is False


# --------------------------------------------------------------------------
# unshorten()
# --------------------------------------------------------------------------


def _fake_getaddrinfo(host_to_ip: dict[str, str]):
    def _getaddrinfo(host, port, *args, **kwargs):
        if host not in host_to_ip:
            raise socket.gaierror(f"no mapping configured for {host!r}")
        ip = host_to_ip[host]
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, port))]

    return _getaddrinfo


@responses.activate
def test_unshorten_follows_safe_redirect_chain():
    responses.add(
        responses.HEAD, "http://short.example/abc", status=301, headers={"Location": "http://mid.example/y"}
    )
    responses.add(responses.HEAD, "http://mid.example/y", status=200)

    with patch(
        "spamdet.preprocessing.url_tools.socket.getaddrinfo",
        side_effect=_fake_getaddrinfo({"short.example": "8.8.8.8", "mid.example": "8.8.4.4"}),
    ):
        result = unshorten("http://short.example/abc")

    assert result.status == "ok"
    assert result.final_url == "http://mid.example/y"
    assert result.redirect_chain == ["http://short.example/abc", "http://mid.example/y"]


@responses.activate
def test_unshorten_blocks_hop_resolving_to_internal_ip():
    responses.add(
        responses.HEAD,
        "http://short.example/abc",
        status=301,
        headers={"Location": "http://internal.example/secret"},
    )
    # No response registered for internal.example - if the SSRF guard did
    # not block it, `responses` would raise for the unmocked call.

    with patch(
        "spamdet.preprocessing.url_tools.socket.getaddrinfo",
        side_effect=_fake_getaddrinfo({"short.example": "8.8.8.8", "internal.example": "169.254.169.254"}),
    ):
        result = unshorten("http://short.example/abc")

    assert result.status == "blocked_ssrf"
    assert "internal.example" in result.error


def test_unshorten_blocks_disallowed_scheme_without_network_call():
    result = unshorten("file:///etc/passwd")
    assert result.status == "blocked_ssrf"


def test_unshorten_blocks_userinfo_in_url():
    with patch(
        "spamdet.preprocessing.url_tools.socket.getaddrinfo",
        side_effect=_fake_getaddrinfo({"evil.example": "8.8.8.8"}),
    ):
        result = unshorten("http://user:pass@evil.example/x")
    assert result.status == "blocked_ssrf"


@responses.activate
def test_unshorten_too_many_redirects():
    for i in range(DEFAULT_MAX_REDIRECTS + 2):
        responses.add(
            responses.HEAD,
            f"http://loop.example/{i}",
            status=301,
            headers={"Location": f"http://loop.example/{i + 1}"},
        )

    with patch(
        "spamdet.preprocessing.url_tools.socket.getaddrinfo",
        side_effect=_fake_getaddrinfo({"loop.example": "8.8.8.8"}),
    ):
        result = unshorten("http://loop.example/0")

    assert result.status == "too_many_redirects"


@responses.activate
def test_unshorten_falls_back_to_get_when_head_not_allowed():
    responses.add(responses.HEAD, "http://noheadmethod.example/x", status=405)
    responses.add(responses.GET, "http://noheadmethod.example/x", status=200)

    with patch(
        "spamdet.preprocessing.url_tools.socket.getaddrinfo",
        side_effect=_fake_getaddrinfo({"noheadmethod.example": "8.8.8.8"}),
    ):
        result = unshorten("http://noheadmethod.example/x")

    assert result.status == "ok"
    assert result.final_url == "http://noheadmethod.example/x"


def test_unshorten_dns_failure_is_blocked_not_crash():
    with patch(
        "spamdet.preprocessing.url_tools.socket.getaddrinfo",
        side_effect=_fake_getaddrinfo({}),
    ):
        result = unshorten("http://doesnotresolve.example/x")
    assert result.status == "blocked_ssrf"
