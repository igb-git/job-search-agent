"""Tencent careers (careers.tencent.com), read through the JSON API behind its search page.

The list response gives each post's title, city, country and a short responsibilities
text. It does not include the full requirements, so scoring works from that summary.
"""

from __future__ import annotations

import json
import time
from urllib.parse import quote, urlencode

from ..fetcher import Fetcher
from .base import AdapterError, ApiJob

ENDPOINT = "https://careers.tencent.com/tencentcareer/api/post/Query"
PAGE_SIZE = 50
MAX_PAGES = 10  # safety limit
JSON_HEADERS = {"Accept": "application/json"}

# (label shown to the LLM, field in the API response)
INFO_FIELDS = (
    ("Business group", "BGName"),
    ("Product", "ProductName"),
    ("Category", "CategoryName"),
    ("Experience required", "RequireWorkYearsName"),
)
TEXT_FIELDS = (("Responsibilities", "Responsibility"), ("Requirements", "Requirement"))


def _request_url(keyword: str, page: int) -> str:
    params = {
        "timestamp": int(time.time() * 1000),
        "countryId": "", "cityId": "", "bgIds": "", "productId": "",
        "categoryId": "", "parentCategoryId": "", "attrId": "",
        "keyword": keyword,
        "pageIndex": page,
        "pageSize": PAGE_SIZE,
        "language": "en-us",
        "area": "us",
    }
    return f"{ENDPOINT}?{urlencode(params, quote_via=quote)}"


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _location(item: dict) -> str | None:
    city, country = _text(item.get("LocationName")), _text(item.get("CountryName"))
    parts = [city] if city else []
    if country and country.lower() != city.lower():
        parts.append(country)
    return ", ".join(parts) or None


def _description(item: dict) -> str | None:
    lines = []
    for label, key in INFO_FIELDS:
        value = _text(item.get(key))
        if value:
            lines.append(f"{label}: {value}")
    for label, key in TEXT_FIELDS:
        value = _text(item.get(key)).replace("\r\n", "\n").replace("\r", "\n")
        if value:
            lines.append(f"\n{label}:\n{value}")
    return "\n".join(lines).strip() or None


def _parse_job(raw: object) -> ApiJob | None:
    if not isinstance(raw, dict):
        return None
    post_id, title = raw.get("PostId"), _text(raw.get("RecruitPostName"))
    post_id = str(post_id).strip() if post_id is not None else ""
    if not post_id or not title:
        return None
    if raw.get("IsValid") is False:
        return None
    return ApiJob(
        id=post_id,
        title=title,
        location=_location(raw),
        description=_description(raw),
    )


def _get_page(fetcher: Fetcher, keyword: str, page: int) -> dict:
    result = fetcher.fetch(_request_url(keyword, page), headers=JSON_HEADERS)
    try:
        body = json.loads(result.text)
    except ValueError:
        raise AdapterError(
            "Tencent returned something that is not JSON. The site may have changed."
        ) from None
    data = body.get("Data") if isinstance(body, dict) else None
    if not isinstance(data, dict):
        code = body.get("Code") if isinstance(body, dict) else None
        raise AdapterError(f"Tencent returned an unexpected response (code: {code}).")
    return data


def search(fetcher: Fetcher, keyword: str) -> list[ApiJob]:
    """All valid posts matching a keyword."""
    found: dict[str, ApiJob] = {}
    for page in range(1, MAX_PAGES + 1):
        data = _get_page(fetcher, keyword, page)
        posts = data.get("Posts") or []
        if not isinstance(posts, list) or not posts:
            break

        before = len(found)
        for raw in posts:
            job = _parse_job(raw)
            if job:
                found.setdefault(job.id, job)
        if len(found) == before:  # a page with nothing new: stop, never loop
            break

        total = data.get("Count")
        if isinstance(total, int) and len(found) >= total:
            break
    return list(found.values())
