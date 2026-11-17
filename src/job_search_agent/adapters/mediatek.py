"""MediaTek careers (careers.mediatek.com), read through the JSON API the site's
own search page uses. The page itself is JavaScript-rendered, so its HTML has no jobs.
"""

from __future__ import annotations

import json
from urllib.parse import quote

from ..fetcher import Fetcher
from .base import AdapterError, ApiJob

ENDPOINT = "https://careers.mediatek.com/api/trpc/job.getJobs"
PAGE_SIZE = 50
MAX_PAGES = 10  # safety limit
JSON_HEADERS = {"Accept": "application/json"}


def _request_url(keyword: str, page: int) -> str:
    payload = {
        "0": {
            "json": {
                "locales": "en_US",
                "page": page,
                "jobQueryInfo": {"keywords": [keyword], "relation": "AND"},
                "filters": {"categorys": [], "workExperiences": [], "locations": [], "programs": []},
                "sortBy": "publishedDate",
                "order": "DESC",
                "limit": PAGE_SIZE,
            }
        }
    }
    compact = json.dumps(payload, separators=(",", ":"))
    return f"{ENDPOINT}?batch=1&input={quote(compact, safe='')}"


def _location(raw: dict) -> str | None:
    """The city, or None. The API stores it under properties.location, and its
    'label' and 'code' fields look swapped (code holds the name), so the first one
    containing letters is used."""
    top = raw.get("location")
    if isinstance(top, str) and top.strip():
        return top.strip()
    properties = raw.get("properties")
    place = properties.get("location") if isinstance(properties, dict) else None
    if isinstance(place, dict):
        for key in ("code", "label"):
            value = place.get(key)
            if isinstance(value, str) and any(ch.isalpha() for ch in value):
                return " ".join(value.split())
    return None


def _parse_job(raw: object) -> ApiJob | None:
    if not isinstance(raw, dict):
        return None
    job_id, title = raw.get("id"), raw.get("title")
    if not isinstance(job_id, str) or not job_id.strip():
        return None
    if not isinstance(title, str) or not title.strip():
        return None
    status = raw.get("jobPostStatus")
    if isinstance(status, str) and status.lower() != "posted":
        return None
    description = raw.get("description")
    return ApiJob(
        id=job_id.strip(),
        title=title.strip(),
        location=_location(raw),
        description=description if isinstance(description, str) else None,
    )


def _get_page(fetcher: Fetcher, keyword: str, page: int) -> dict:
    result = fetcher.fetch(_request_url(keyword, page), headers=JSON_HEADERS)
    try:
        body = json.loads(result.text)
    except ValueError:
        raise AdapterError(
            "MediaTek returned something that is not JSON. The site may have changed."
        ) from None
    try:
        first = body[0]
    except (KeyError, IndexError, TypeError):
        raise AdapterError("MediaTek returned an unexpected response shape.") from None
    if isinstance(first, dict) and "error" in first:
        raise AdapterError(f"MediaTek API error: {json.dumps(first['error'])[:200]}")
    try:
        data = first["result"]["data"]["json"]
    except (KeyError, TypeError):
        raise AdapterError("MediaTek returned an unexpected response shape.") from None
    if not isinstance(data, dict):
        raise AdapterError("MediaTek returned an unexpected response shape.")
    return data


def search(fetcher: Fetcher, keyword: str) -> list[ApiJob]:
    """All posted jobs matching a keyword, newest first."""
    found: dict[str, ApiJob] = {}
    for page in range(1, MAX_PAGES + 1):
        data = _get_page(fetcher, keyword, page)
        raw_jobs = data.get("jobs") or []
        if not raw_jobs:
            break

        before = len(found)
        for raw in raw_jobs:
            job = _parse_job(raw)
            if job:
                found.setdefault(job.id, job)
        if len(found) == before:  # a page with nothing new: stop, never loop
            break

        pagination = data.get("pagination")
        total = pagination.get("total_items") if isinstance(pagination, dict) else None
        if isinstance(total, int) and len(found) >= total:
            break
    return list(found.values())
