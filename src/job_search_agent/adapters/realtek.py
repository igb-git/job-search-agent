"""Realtek careers API adapter."""

from __future__ import annotations

import json

from ..fetcher import Fetcher
from .base import AdapterError, ApiJob

ENDPOINT = "https://recruit.realtek.com/Job/GetFilterList"

HEADERS = {
    "Accept": "application/json",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://recruit.realtek.com/Job/Search?lang=en",
    "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_job(raw: object) -> ApiJob | None:
    if not isinstance(raw, dict):
        return None

    # JobOppId is the ID used by JobDetail?jobid=
    job_id = _text(raw.get("JobOppId"))
    title = _text(raw.get("JobTitle"))

    if not job_id or not title:
        return None

    location = _text(raw.get("Location")) or None

    details = []
    for label, field in (
        ("Category", "JobType"),
        ("Education", "Degree"),
        ("Experience", "Exp"),
    ):
        value = _text(raw.get(field))
        if value:
            details.append(f"{label}: {value}")

    return ApiJob(
        id=job_id,
        title=title,
        location=location,
        description="\n".join(details) or None,
    )


def search(fetcher: Fetcher, keyword: str) -> list[ApiJob]:
    """Search Realtek vacancies using its public filtered-job endpoint."""

    payload = {
        "keyword": keyword,
        "job_location": "",
        "xp": "-1",
        "job_type_id": "",
    }

    result = fetcher.fetch(
        ENDPOINT,
        headers=HEADERS,
        method="POST",
        data=payload,
    )

    try:
        body = json.loads(result.text)
    except ValueError:
        raise AdapterError(
            "Realtek returned something that is not JSON."
        ) from None

    if not isinstance(body, dict):
        raise AdapterError("Realtek returned an unexpected response.")

    if body.get("Pass") is not True:
        raise AdapterError("Realtek search request was unsuccessful.")

    raw_jobs = body.get("Data")

    if not isinstance(raw_jobs, list):
        raise AdapterError("Realtek returned an unexpected job list.")

    found: dict[str, ApiJob] = {}

    for raw in raw_jobs:
        job = _parse_job(raw)
        if job:
            found.setdefault(job.id, job)

    return list(found.values())
