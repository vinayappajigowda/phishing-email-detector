"""Rule-based checks on an email: sender authentication, links and wording.

These checks don't decide the verdict on their own. They explain *why* an email
looks suspicious, which is what an analyst needs alongside the model's score.
"""
from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from email import policy
from email.parser import Parser
from email.utils import parseaddr
from html import unescape
from urllib.parse import urlparse

URL_RE = re.compile(r"""(?i)\b((?:https?://|www\.)[^\s<>"')\]]+)""")
HREF_RE = re.compile(r"""(?is)<a\s[^>]*href\s*=\s*["']([^"']+)["'][^>]*>(.*?)</a>""")
TAG_RE = re.compile(r"(?s)<[^>]+>")

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "rb.gy", "tiny.cc",
}
RISKY_TLDS = {
    "zip", "mov", "xyz", "top", "click", "country", "gq", "tk", "ml", "cf",
    "ga", "work", "support", "rest", "fit", "loan",
}
URGENCY_PHRASES = [
    "verify your account", "account suspended", "account will be closed",
    "unusual activity", "suspicious activity", "confirm your identity",
    "update your payment", "password expires", "act now", "immediately",
    "within 24 hours", "urgent", "final notice", "you have won",
    "click here", "login to restore", "unlock your account",
]


@dataclass
class ParsedEmail:
    subject: str
    sender: str
    reply_to: str
    return_path: str
    auth_results: str
    text: str          # plain text body (HTML stripped)
    html: str          # raw HTML body, if any
    has_headers: bool


@dataclass
class Flag:
    severity: str      # "high", "medium" or "low"
    category: str      # "sender", "link" or "content"
    message: str


def _domain(address: str) -> str:
    _, addr = parseaddr(address or "")
    return addr.rsplit("@", 1)[-1].lower().strip(">") if "@" in addr else ""


def _looks_like_headers(raw: str) -> bool:
    head = raw.lstrip()[:2000]
    return bool(re.match(r"(?im)^(from|received|subject|return-path|delivered-to|date|to):", head))


def html_to_text(html: str) -> str:
    return re.sub(r"\s+", " ", unescape(TAG_RE.sub(" ", html))).strip()


def parse_email(raw: str) -> ParsedEmail:
    """Parse a raw .eml file, or plain pasted text with no headers."""
    if not _looks_like_headers(raw):
        return ParsedEmail("", "", "", "", "", raw.strip(), "", False)

    msg = Parser(policy=policy.default).parsestr(raw)
    text_parts, html_parts = [], []
    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = part.get_content_type()
        try:
            content = part.get_content()
        except Exception:  # broken encodings are common in real phishing
            payload = part.get_payload(decode=True) or b""
            content = payload.decode("utf-8", errors="replace")
        if not isinstance(content, str):
            continue
        if ctype == "text/plain":
            text_parts.append(content)
        elif ctype == "text/html":
            html_parts.append(content)

    html = "\n".join(html_parts)
    text = "\n".join(text_parts) or html_to_text(html)
    return ParsedEmail(
        subject=str(msg.get("Subject", "")),
        sender=str(msg.get("From", "")),
        reply_to=str(msg.get("Reply-To", "")),
        return_path=str(msg.get("Return-Path", "")),
        auth_results=" ".join(str(h) for h in msg.get_all("Authentication-Results", [])),
        text=text.strip(),
        html=html,
        has_headers=True,
    )


def extract_urls(text: str, html: str = "") -> list[str]:
    urls = URL_RE.findall(text or "")
    urls += [href for href, _ in HREF_RE.findall(html or "")]
    seen, out = set(), []
    for u in urls:
        u = u.rstrip(".,;:!?")
        if u.lower().startswith(("http", "www.")) and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _host(url: str) -> str:
    if url.lower().startswith("www."):
        url = "http://" + url
    return (urlparse(url).hostname or "").lower()


def url_flags(url: str) -> list[Flag]:
    flags: list[Flag] = []
    host = _host(url)
    if not host:
        return flags
    short = url if len(url) <= 60 else url[:57] + "..."
    try:
        ipaddress.ip_address(host)
        flags.append(Flag("high", "link", f"Link points to a raw IP address: {short}"))
    except ValueError:
        pass
    if host in SHORTENERS:
        flags.append(Flag("medium", "link", f"Shortened link hides the real destination: {short}"))
    if "@" in urlparse(url if "://" in url else "http://" + url).netloc:
        flags.append(Flag("high", "link", f"Link uses '@' to disguise the real host: {short}"))
    if "xn--" in host:
        flags.append(Flag("high", "link", f"Link uses a look-alike (punycode) domain: {short}"))
    if host.rsplit(".", 1)[-1] in RISKY_TLDS:
        flags.append(Flag("medium", "link", f"Link uses a high-risk domain ending (.{host.rsplit('.', 1)[-1]}): {short}"))
    if host.count(".") >= 4:
        flags.append(Flag("medium", "link", f"Link has an unusually long chain of subdomains: {short}"))
    if url.lower().startswith("http://") and re.search(r"(?i)login|sign-?in|verify|account|password", url):
        flags.append(Flag("medium", "link", f"Login-style link without HTTPS: {short}"))
    return flags


def mismatched_links(html: str) -> list[Flag]:
    """Visible link text shows one domain but the link goes somewhere else."""
    flags = []
    for href, label in HREF_RE.findall(html or ""):
        shown = html_to_text(label)
        shown_url = URL_RE.search(shown)
        if not shown_url:
            continue
        shown_host, real_host = _host(shown_url.group(1)), _host(href)
        if shown_host and real_host and not real_host.endswith(shown_host.removeprefix("www.")):
            flags.append(Flag("high", "link",
                              f"Link text says {shown_host} but actually goes to {real_host}"))
    return flags


def auth_flags(auth_results: str) -> list[Flag]:
    flags = []
    if not auth_results:
        return flags
    for check in ("spf", "dkim", "dmarc"):
        m = re.search(rf"(?i)\b{check}=(\w+)", auth_results)
        if m and m.group(1).lower() in {"fail", "softfail", "none", "permerror", "temperror"}:
            sev = "high" if m.group(1).lower() in {"fail", "softfail"} else "medium"
            flags.append(Flag(sev, "sender", f"{check.upper()} check result: {m.group(1).lower()}"))
    return flags


def sender_flags(email: ParsedEmail) -> list[Flag]:
    flags = auth_flags(email.auth_results)
    from_dom = _domain(email.sender)
    reply_dom = _domain(email.reply_to)
    path_dom = _domain(email.return_path)
    if from_dom and reply_dom and reply_dom != from_dom:
        flags.append(Flag("high", "sender",
                          f"Replies go to a different domain ({reply_dom}) than the sender ({from_dom})"))
    if from_dom and path_dom and path_dom != from_dom and not path_dom.endswith("." + from_dom):
        flags.append(Flag("medium", "sender",
                          f"Return-Path domain ({path_dom}) doesn't match the sender ({from_dom})"))
    return flags


def content_flags(subject: str, text: str) -> list[Flag]:
    blob = f"{subject} {text}".lower()
    hits = [p for p in URGENCY_PHRASES if p in blob]
    if not hits:
        return []
    sev = "medium" if len(hits) >= 2 else "low"
    return [Flag(sev, "content", "Pressure or urgency wording: " + ", ".join(f"'{h}'" for h in hits[:5]))]


def red_flags(raw: str) -> tuple[ParsedEmail, list[str], list[Flag]]:
    """Run every rule-based check. Returns the parsed email, its URLs and the flags."""
    email = parse_email(raw)
    urls = extract_urls(email.text, email.html)
    flags = sender_flags(email) if email.has_headers else []
    flags += mismatched_links(email.html)
    for u in urls:
        flags += url_flags(u)
    flags += content_flags(email.subject, email.text)
    order = {"high": 0, "medium": 1, "low": 2}
    flags.sort(key=lambda f: order[f.severity])
    return email, urls, flags
