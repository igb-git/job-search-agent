"""Shared pieces for API sources (companies whose job list comes from a JSON API).

An adapter is a function `search(fetcher, keyword) -> list[ApiJob]`. It knows how
to ask one company's API for jobs matching a keyword and how to read the answer.
Everything after that (filters, scoring, Telegram) is shared with the HTML sources.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..pagetext import DEFAULT_MAX_CHARS
from ..pagetext import _html_to_text as html_to_text  # converts HTML fragments to clean text

__all__ = ["AdapterError", "ApiJob", "html_to_text", "job_text"]


class AdapterError(Exception):
    """The API answered, but not in the shape the adapter expects."""


@dataclass(frozen=True)
class ApiJob:
    id: str  # the company's own job ID, used to build the posting URL
    title: str
    location: str | None
    description: str | None


def job_text(job: ApiJob, company: str) -> str:
    """The text sent to the LLM, in the same layout pagetext.py produces for web pages."""
    header = [f"Title: {job.title}", f"Company: {company}"]
    if job.location:
        header.append(f"Location: {job.location}")
    body = (job.description or "").strip()
    text = "\n".join(header) + ("\n\n" + body if body else "")
    return text[:DEFAULT_MAX_CHARS]
