"""Parses sources.txt and builds one search URL per (source, keyword) pair.

sources.txt format, one source per line:

    name | url_template
    name | url_template | link_pattern
    name | adapter:<adapter_name> | job_url_template

The last form is for companies read through a JSON API (see adapters/). Its
job_url_template must contain {id}, which is replaced by the job's ID to give the
posting's public address.

The template must contain the {query} placeholder, which is replaced by each
keyword. The optional link_pattern is a regular expression that job posting
URLs match on that site; it lets the agent pick out job links without the LLM.
Lines starting with # and blank lines are ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, quote_plus, urlsplit

from .adapters import ADAPTERS

PLACEHOLDER = "{query}"


class SourcesError(Exception):
    """Raised when sources.txt is missing, unreadable or invalid."""


@dataclass(frozen=True)
class Source:
    name: str
    url_template: str
    line_number: int
    link_pattern: str | None = None  # regex that job posting URLs match, if known
    adapter: str | None = None  # name of a JSON API adapter, for API sources
    job_url: str | None = None  # posting URL template with {id}, for API sources


@dataclass(frozen=True)
class SearchTarget:
    """One results page to download: a source searched for one keyword."""

    source_name: str
    keyword: str
    url: str
    link_pattern: str | None = None  # copied from the source line


def parse_sources(text: str) -> list[Source]:
    """Parse the content of a sources file. Raises SourcesError listing every bad line."""
    sources: list[Source] = []
    errors: list[str] = []
    seen_templates: set[str] = set()

    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if "|" not in line:
            errors.append(f"line {line_number}: expected 'name | url_template'.")
            continue

        # Split on the first two pipes only. URLs do not contain pipes, and the
        # pattern may contain them (regex alternation), so it keeps the rest.
        name, _, rest = line.partition("|")
        template, _, pattern = rest.partition("|")
        name = name.strip()
        template = template.strip()
        pattern = pattern.strip() or None

        if not name:
            errors.append(f"line {line_number}: the name before '|' is empty.")
            continue
        if template.lower().startswith("adapter:"):
            adapter = template[len("adapter:"):].strip().lower()
            if adapter not in ADAPTERS:
                errors.append(
                    f"line {line_number} ({name}): unknown adapter '{adapter}'. "
                    f"Available: {', '.join(sorted(ADAPTERS))}."
                )
                continue
            job_url = pattern or ""
            job_parts = urlsplit(job_url.replace("{id}", "x"))
            if (
                "{id}" not in job_url
                or any(ch.isspace() for ch in job_url)
                or job_parts.scheme not in ("http", "https")
                or not job_parts.netloc
            ):
                errors.append(
                    f"line {line_number} ({name}): an adapter line needs a third field with "
                    "the posting URL containing {id}, for example https://example.com/jobs/{id}."
                )
                continue
            sources.append(
                Source(name=name, url_template="", line_number=line_number,
                       adapter=adapter, job_url=job_url)
            )
            continue

        if PLACEHOLDER not in template:
            errors.append(
                f"line {line_number} ({name}): the URL has no {PLACEHOLDER} placeholder."
            )
            continue
        if any(ch.isspace() for ch in template):
            errors.append(
                f"line {line_number} ({name}): the URL contains spaces. "
                "Put notes on a separate '#' comment line instead."
            )
            continue

        parts = urlsplit(template.replace(PLACEHOLDER, "x"))
        if parts.scheme not in ("http", "https") or not parts.netloc:
            errors.append(
                f"line {line_number} ({name}): the URL must start with http:// or https://."
            )
            continue

        if pattern is not None:
            try:
                re.compile(pattern)
            except re.error as exc:
                errors.append(
                    f"line {line_number} ({name}): the link pattern is not a valid "
                    f"regular expression ({exc})."
                )
                continue

        if template in seen_templates:
            continue  # exact duplicate line, ignore silently
        seen_templates.add(template)

        sources.append(
            Source(
                name=name,
                url_template=template,
                line_number=line_number,
                link_pattern=pattern,
            )
        )

    if errors:
        bullet_list = "\n".join(f"  - {e}" for e in errors)
        raise SourcesError(f"Problems in the sources file:\n{bullet_list}")
    return sources


def load_sources(path: str | Path) -> list[Source]:
    """Read and parse the sources file."""
    path = Path(path)
    if not path.is_file():
        raise SourcesError(
            f"{path} not found. Copy sources.example.txt to sources.txt first."
        )
    try:
        # utf-8-sig silently drops a BOM, which some editors add at the start.
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise SourcesError(f"Cannot read {path}: {exc}") from exc

    sources = parse_sources(text)
    if not sources:
        raise SourcesError(f"{path} contains no sources (only comments or blank lines).")
    return sources


def fill_template(template: str, keyword: str) -> str:
    """Insert a keyword into a URL template, encoded for where the placeholder sits.

    After a '?' (query string) spaces become '+'. In the path they become '%20'.
    Each occurrence of the placeholder is encoded for its own position.
    """
    pieces = template.split(PLACEHOLDER)
    result = pieces[0]
    for piece in pieces[1:]:
        in_query = "?" in result
        encoded = quote_plus(keyword) if in_query else quote(keyword, safe="")
        result += encoded + piece
    return result


def build_search_targets(sources: list[Source], keywords: list[str]) -> list[SearchTarget]:
    """Combine every source with every keyword, skipping duplicate URLs."""
    targets: list[SearchTarget] = []
    seen_urls: set[str] = set()

    for source in sources:
        if source.adapter:  # API sources are queried by the pipeline, not downloaded
            continue
        for keyword in keywords:
            keyword = keyword.strip()
            if not keyword:
                continue
            url = fill_template(source.url_template, keyword)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            targets.append(
                SearchTarget(
                    source_name=source.name,
                    keyword=keyword,
                    url=url,
                    link_pattern=source.link_pattern,
                )
            )
    return targets
