"""Scores a job posting against the candidate's profile with an LLM.

The prompt lives in prompts/score_job.txt. This module fills it in, sends it,
and checks that the answer is usable before anything is stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .llm.base import LLMClient, LLMError

PROMPT_PATH = Path(__file__).with_name("prompts") / "score_job.txt"
SYSTEM_PROMPT = (
    "You are a careful job-matching assistant. "
    "You reply only with a single JSON object and nothing else."
)
EMPTY_VALUES = {"", "null", "none", "unknown", "n/a", "not specified"}
MAX_REASON_CHARS = 300


@dataclass(frozen=True)
class MatchResult:
    title: str | None
    company: str | None
    location: str | None
    country: str | None
    tier: str | None
    score: float  # 0 to 10
    reason: str


def load_prompt_template(path: Path = PROMPT_PATH) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LLMError(f"Cannot read the scoring prompt {path}: {exc}") from exc


def build_prompt(
    template: str, profile: str, blocked_countries: list[str], job_text: str
) -> str:
    """Fill in the template. Plain replacement, because the template contains braces."""
    blocked = ", ".join(blocked_countries) if blocked_countries else "none"
    return (
        template.replace("{profile}", profile.strip())
        .replace("{blocked_countries}", blocked)
        .replace("{job_text}", job_text.strip())
    )


def _optional_text(value: object) -> str | None:
    """Clean a text field. Empty strings and 'null'/'unknown' become None."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in EMPTY_VALUES else text


def parse_result(data: dict) -> MatchResult:
    """Validate the model's JSON and turn it into a MatchResult.

    Raises LLMError if the score or the reason is missing or unusable. Other
    fields are cleaned up leniently, since small models are inconsistent there.
    """
    raw_score = data.get("score")
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        raise LLMError(f"The model returned an invalid score: {raw_score!r}") from None
    if not 0 <= score <= 10:  # also rejects NaN
        raise LLMError(f"The model returned a score outside 0-10: {raw_score!r}")

    reason = _optional_text(data.get("reason"))
    if reason is None:
        raise LLMError("The model returned no reason.")

    tier = _optional_text(data.get("tier"))
    return MatchResult(
        title=_optional_text(data.get("title")),
        company=_optional_text(data.get("company")),
        location=_optional_text(data.get("location")),
        country=_optional_text(data.get("country")),
        tier=tier[:30] if tier else None,
        score=score,
        reason=reason[:MAX_REASON_CHARS],
    )


def score_job(
    client: LLMClient,
    profile: str,
    blocked_countries: list[str],
    job_text: str,
    template: str | None = None,
) -> MatchResult:
    """Score one job. Raises LLMError if the model fails or answers unusably."""
    if template is None:
        template = load_prompt_template()
    prompt = build_prompt(template, profile, blocked_countries, job_text)
    data = client.complete_json(SYSTEM_PROMPT, prompt)
    return parse_result(data)
