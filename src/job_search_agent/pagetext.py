"""Reduces a downloaded job posting page to the text the LLM should read.

Two sources of information are combined:

1. Structured data. Many career sites embed a schema.org "JobPosting" block
   (JSON-LD) in the page for search engines. When present it gives a clean
   title, company, location and description, so it is used first.
2. The visible page text, with scripts, menus, footers and cookie banners
   removed, as a fallback and when the structured description is missing.
"""

from __future__ import annotations

import html as html_lib
import json
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

DEFAULT_MAX_CHARS = 12_000  # roughly 3,000 tokens, comfortable for a 14B local model
MIN_USEFUL_CHARS = 200  # below this the page is probably an empty JavaScript shell
MIN_STRUCTURED_DESCRIPTION = 200

NOISE_TAGS = ["script", "style", "noscript", "svg", "iframe", "nav", "footer",
              "aside", "form", "button", "select"]
NOISE_ATTRS = re.compile(r"cookie|consent|gdpr|onetrust", re.IGNORECASE)
NEVER_REMOVE = {"html", "body", "main", "article"}


@dataclass(frozen=True)
class PageText:
    title: str | None
    text: str  # what to send to the LLM: a short header followed by the body
    has_structured_data: bool  # True if a JobPosting block was found
    truncated: bool

    @property
    def looks_empty(self) -> bool:
        """True when almost nothing was extracted, e.g. a JavaScript-rendered page."""
        return len(self.text) < MIN_USEFUL_CHARS


def _as_text(value: object) -> str:
    """Coerce a JSON-LD value (string, or object with a name) to plain text."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return _as_text(value.get("name", ""))
    return ""


def _html_to_text(fragment: str) -> str:
    if "&lt;" in fragment:  # some sites double-escape the HTML in the description
        fragment = html_lib.unescape(fragment)
    soup = BeautifulSoup(fragment, "lxml")
    return _clean_lines(soup.get_text("\n"))


def _clean_lines(text: str) -> str:
    """Strip each line, drop empty lines and consecutive duplicates."""
    lines: list[str] = []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if line and (not lines or lines[-1] != line):
            lines.append(line)
    return "\n".join(lines)


def _find_job_posting(soup: BeautifulSoup) -> dict | None:
    """Return the first schema.org JobPosting object found in the page's JSON-LD."""

    def search(node: object) -> dict | None:
        if isinstance(node, list):
            for item in node:
                found = search(item)
                if found:
                    return found
        elif isinstance(node, dict):
            kind = node.get("@type")
            kinds = kind if isinstance(kind, list) else [kind]
            if "JobPosting" in kinds:
                return node
            return search(node.get("@graph"))
        return None

    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text()
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        found = search(data)
        if found:
            return found
    return None


def _format_location(job: dict) -> str:
    locations = job.get("jobLocation")
    if isinstance(locations, dict):
        locations = [locations]
    places: list[str] = []
    for location in locations or []:
        if not isinstance(location, dict):
            continue
        address = location.get("address", {})
        if isinstance(address, str):
            places.append(address.strip())
            continue
        if not isinstance(address, dict):
            continue
        parts = [
            _as_text(address.get("addressLocality")),
            _as_text(address.get("addressRegion")),
            _as_text(address.get("addressCountry")),
        ]
        place = ", ".join(part for part in parts if part)
        if place:
            places.append(place)
    return "; ".join(places)


def _structured_header(job: dict) -> tuple[str | None, list[str]]:
    """Build the header lines and the title from a JobPosting object."""
    title = _as_text(job.get("title")) or None
    lines: list[str] = []
    if title:
        lines.append(f"Title: {title}")
    company = _as_text(job.get("hiringOrganization"))
    if company:
        lines.append(f"Company: {company}")
    location = _format_location(job)
    if location:
        lines.append(f"Location: {location}")
    if str(job.get("jobLocationType", "")).upper() == "TELECOMMUTE":
        lines.append("Remote: yes")
    employment = job.get("employmentType")
    if isinstance(employment, list):
        employment = ", ".join(str(e) for e in employment)
    if employment:
        lines.append(f"Employment type: {employment}")
    return title, lines


def _remove_noise(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(NOISE_TAGS):
        tag.decompose()

    # Cookie banners and consent popups, found by id or class name.
    def is_banner(tag: Tag) -> bool:
        if tag.name in NEVER_REMOVE:
            return False
        classes = " ".join(tag.get("class") or [])
        return bool(NOISE_ATTRS.search(f"{tag.get('id') or ''} {classes}"))

    for tag in soup.find_all(is_banner):
        if not getattr(tag, "decomposed", False):
            tag.decompose()


def _main_text(soup: BeautifulSoup) -> str:
    """Text of the page's main content area, or of the whole body as a fallback."""
    candidates = soup.find_all(["main", "article"]) + soup.find_all(attrs={"role": "main"})
    best = ""
    for candidate in candidates:
        text = _clean_lines(candidate.get_text("\n"))
        if len(text) > len(best):
            best = text
    if len(best) >= 300:
        return best
    body = soup.body or soup
    return _clean_lines(body.get_text("\n"))


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    cut = text[:max_chars]
    newline = cut.rfind("\n")
    if newline > max_chars * 0.8:  # prefer cutting at a line boundary
        cut = cut[:newline]
    return cut + "\n[text truncated]", True


def extract_page_text(html: str, max_chars: int = DEFAULT_MAX_CHARS) -> PageText:
    """Turn a job posting page into a PageText ready to send to the LLM."""
    soup = BeautifulSoup(html, "lxml")

    # Structured data must be read before scripts are removed.
    job = _find_job_posting(soup)
    title: str | None = None
    header: list[str] = []
    description = ""
    if job:
        title, header = _structured_header(job)
        raw_description = job.get("description")
        if isinstance(raw_description, str):
            description = _html_to_text(raw_description)

    if not title:
        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            title = og["content"].strip()
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()

    if len(description) >= MIN_STRUCTURED_DESCRIPTION:
        body = description
    else:
        _remove_noise(soup)
        body = _main_text(soup)

    parts = ["\n".join(header), body] if header else [body]
    text, truncated = _truncate("\n\n".join(parts), max_chars)
    return PageText(
        title=title,
        text=text,
        has_structured_data=job is not None,
        truncated=truncated,
    )
