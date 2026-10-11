from pathlib import Path

from src.data import clean_text
from src.features import extract_urls, parse_email, red_flags, url_flags

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def _messages(flags):
    return " | ".join(f.message for f in flags)


def test_phishing_sample_raises_sender_and_link_flags():
    email, urls, flags = red_flags((SAMPLES / "phishing_example.eml").read_text())
    msgs = _messages(flags)
    assert email.has_headers
    assert "SPF check result: fail" in msgs
    assert "DMARC check result: fail" in msgs
    assert "Replies go to a different domain" in msgs
    assert "raw IP address" in msgs
    assert "Shortened link" in msgs
    assert "Link text says www.northbank.ie but actually goes to 203.0.113.45" in msgs
    assert flags[0].severity == "high"          # sorted most severe first
    assert any("203.0.113.45" in u for u in urls)


def test_legitimate_sample_has_no_high_flags():
    _, _, flags = red_flags((SAMPLES / "legitimate_example.eml").read_text())
    assert not [f for f in flags if f.severity == "high"]


def test_plain_text_without_headers():
    email = parse_email("Hello, please review the attached report before Friday.")
    assert not email.has_headers
    assert email.text.startswith("Hello")


def test_url_checks():
    assert not url_flags("https://www.example.org/news")
    assert any("@" in f.message for f in url_flags("http://example.org@evil.top/login"))
    assert any("punycode" in f.message for f in url_flags("https://xn--pypal-4ve.com/"))
    assert extract_urls("see https://a.example.com/x, and www.b.example.org.") == [
        "https://a.example.com/x", "www.b.example.org"]


def test_clean_text_replaces_urls_emails_numbers():
    out = clean_text("Pay 250.00 EUR at https://x.top/pay or mail bob@x.top NOW")
    assert out == "pay numtoken eur at urltoken or mail emailtoken now"
