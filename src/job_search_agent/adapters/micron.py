"""Micron careers (micron.wd1.myworkdayjobs.com), read through the MyWorkday JSON API.

The careers site uses Workday's standard API endpoint for job listings.
"""

from __future__ import annotations

import json

from ..fetcher import Fetcher
from .base import AdapterError, ApiJob, html_to_text

ENDPOINT = "https://micron.wd1.myworkdayjobs.com/wday/cxs/micron/External/jobs"
PAGE_SIZE = 20  # Workday max
MAX_PAGES = 10  # safety limit
JSON_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Referer": "https://micron.wd1.myworkdayjobs.com/wday/cxs/micron/External/home",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
}


def search(fetcher: Fetcher, keyword: str) -> list[ApiJob]:
    """Fetch Micron jobs matching the keyword.

    Args:
        fetcher: HTTP client with timeout/retry/throttle
        keyword: Search term (passed to Workday's searchText field)

    Returns:
        List of ApiJob objects for matching postings

    Raises:
        AdapterError: If the API returns an unexpected response
    """
    jobs: list[ApiJob] = []

    # Fetch up to MAX_PAGES to be safe
    for page in range(MAX_PAGES):
        body = {
            "appliedFacets": {},
            "limit": PAGE_SIZE,
            "offset": page * PAGE_SIZE,
            "searchText": keyword,
        }
        compact = json.dumps(body, separators=(",", ":"))

        response = fetcher.fetch(
            ENDPOINT,
            headers=JSON_HEADERS,
            method="POST",
            json_body=compact,
        )

        if response.status_code != 200:
            raise AdapterError(
                f"Workday API returned status {response.status_code}"
            )

        try:
            data = json.loads(response.text)
        except json.JSONDecodeError as e:
            raise AdapterError(f"Workday API returned invalid JSON: {e}")

        # Validate response shape
        if not isinstance(data, dict):
            raise AdapterError(f"Workday API returned non-object JSON: {type(data)}")

        total = data.get("total", 0)
        job_postings = data.get("jobPostings", [])

        if not job_postings and page == 0:
            # No jobs on first page means no results
            break

        for jp in job_postings:
            title = jp.get("title", "")
            location = jp.get("locationsText")
            req_id = jp.get("bulletFields", [""])[0] if jp.get("bulletFields") else ""

            # Workday list endpoint doesn't provide descriptions; leave empty
            # Users can visit the job page at micron.wd1.myworkdayjobs.com for full details
            description = None

            jobs.append(
                ApiJob(
                    id=req_id,
                    title=title,
                    location=location,
                    description=description,
                )
            )

        # If we've retrieved all jobs or got fewer than page size, we're done
        if len(job_postings) < PAGE_SIZE:
            break
        if total <= (page + 1) * PAGE_SIZE:
            break

    return jobs
