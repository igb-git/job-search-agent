"""Extracts links from a results page.

Links are extracted in code, never by the LLM, so URLs are always real. A
later step decides which of them are job postings, either with a URL pattern
learned for the company or with the LLM as a fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urldefrag, urljoin, urlsplit

from bs4 import BeautifulSoup

IGNORED_PREFIXES = ("mailto:", "tel:", "javascript:", "data:")


@dataclass(frozen=True)
class Link:
    url: str  # absolute, without #fragment
    text: str  # visible link text, whitespace collapsed (may be empty)


def extract_links(html: str, base_url: str) -> list[Link]:
    """Return every distinct link on the page as an absolute URL, in page order.

    base_url should be the final URL of the page (after redirects), so relative
    links such as /jobs/123 resolve correctly. If the same URL appears several
    times, the first non-empty link text is kept.
    """
    soup = BeautifulSoup(html, "lxml")

    # A <base href> tag changes how relative links resolve.
    base_tag = soup.find("base", href=True)
    if base_tag:
        base_url = urljoin(base_url, base_tag["href"].strip())

    texts: dict[str, str] = {}  # dicts keep insertion order
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href or href.startswith("#") or href.lower().startswith(IGNORED_PREFIXES):
            continue

        url, _fragment = urldefrag(urljoin(base_url, href))
        if urlsplit(url).scheme not in ("http", "https"):
            continue

        text = " ".join(anchor.get_text(" ", strip=True).split())
        if url not in texts or (text and not texts[url]):
            texts[url] = text

    return [Link(url=url, text=text) for url, text in texts.items()]


def filter_by_pattern(links: list[Link], pattern: str) -> list[Link]:
    """Keep links whose URL matches a regular expression, e.g. r'/jobs/\\d+-'."""
    regex = re.compile(pattern)
    return [link for link in links if regex.search(link.url)]
