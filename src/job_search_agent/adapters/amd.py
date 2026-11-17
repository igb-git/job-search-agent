"""AMD careers (careers.amd.com), read through the JSON API behind its job search page.

The response holds each job under jobs[i]["data"], with the city, state and
country in separate fields and the job text split over several fields.
"""

from __future__ import annotations

import json
from urllib.parse import quote, urlencode

from ..fetcher import Fetcher
from .base import AdapterError, ApiJob, html_to_text

ENDPOINT = "https://careers.amd.com/api/jobs"
PAGE_SIZE = 100
MAX_PAGES = 10  # safety limit
JSON_HEADERS = {"Accept": "application/json"}
TEXT_FIELDS = ("description", "responsibilities", "qualifications")


def _request_url(keyword: str, page: int) -> str:
    query = urlencode(
        {"keywords": keyword, "sortBy": "relevance", "page": page, "limit": PAGE_SIZE},
        quote_via=quote,
    )
    return f"{ENDPOINT}?{query}"


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _location(item: dict) -> str | None:
    """'City, State, Country'. The country is added when the location text lacks it,
    so the country filter can recognise it ("San Jose, California" has no country)."""
    place = _text(item.get("full_location")) or ", ".join(
        part for part in (_text(item.get("city")), _text(item.get("state"))) if part
    )
    country = _text(item.get("country"))
    if country and country.lower() not in place.lower():
        place = f"{place}, {country}" if place else country
    return " ".join(place.split()) or None


def _description(item: dict) -> str | None:
    """The job text: description, responsibilities and qualifications, whichever
    exist, converted from HTML. A section already contained in another is dropped."""
    sections: list[str] = []
    for key in TEXT_FIELDS:
        value = item.get(key)
        text = html_to_text(value) if isinstance(value, str) and value.strip() else ""
        if not text or any(text in existing for existing in sections):
            continue
        sections = [existing for existing in sections if existing not in text]
        sections.append(text)
    return "\n\n".join(sections) or None


def _parse_job(raw: object) -> ApiJob | None:
    item = raw.get("data", raw) if isinstance(raw, dict) else None
    if not isinstance(item, dict):
        return None

    job_id = item.get("slug") or item.get("req_id")
    job_id = str(job_id).strip() if job_id is not None else ""
    title = _text(item.get("title"))
    if not job_id or not title:
        return None

    location = _location(item)
    description = _description(item)

    if item.get("multipleLocations") is True:
        # Only one location is given, so it must not be used to reject the job.
        # It is passed to the LLM as text instead.
        note = (
            f"Primary location: {location} (this posting lists multiple locations)."
            if location
            else "This posting lists multiple locations."
        )
        description = note + (f"\n\n{description}" if description else "")
        location = None

    return ApiJob(id=job_id, title=title, location=location, description=description)


def _get_page(fetcher: Fetcher, keyword: str, page: int) -> dict:
    result = fetcher.fetch(_request_url(keyword, page), headers=JSON_HEADERS)
    try:
        data = json.loads(result.text)
    except ValueError:
        raise AdapterError(
            "AMD returned something that is not JSON. The site may have changed."
        ) from None
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
        raise AdapterError("AMD returned an unexpected response shape.")
    return data


def search(fetcher: Fetcher, keyword: str) -> list[ApiJob]:
    """All jobs matching a keyword, most relevant first."""
    found: dict[str, ApiJob] = {}
    for page in range(1, MAX_PAGES + 1):
        data = _get_page(fetcher, keyword, page)
        if not data["jobs"]:
            break

        before = len(found)
        for raw in data["jobs"]:
            job = _parse_job(raw)
            if job:
                found.setdefault(job.id, job)
        if len(found) == before:  # a page with nothing new: stop, never loop
            break

        total = data.get("totalCount")
        if isinstance(total, int) and len(found) >= total:
            break
    return list(found.values())
