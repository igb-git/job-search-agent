"""Rule-based filters that run before the LLM, to save time on jobs that are
clearly unwanted.

The country filter is deliberately conservative: it blocks a job only when its
location is clearly in a blocked country. Anything ambiguous passes through,
and the LLM (which is told the excluded countries in the scoring prompt) acts
as the safety net. Every rejection carries a reason so mistakes can be audited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class FilterResult:
    blocked: bool
    reason: str | None = None


PASS = FilterResult(blocked=False)


@dataclass(frozen=True)
class _CountryGroup:
    label: str  # shown in rejection reasons
    names: tuple[str, ...]  # what a user may write in config.yaml (lowercase)
    phrases: tuple[str, ...]  # matched case-insensitively in location text
    tokens: tuple[str, ...] = ()  # matched case-sensitively, e.g. "US"
    state_codes: bool = False  # also match US-style ", TX" endings


# Built-in knowledge for common blocked countries. Any other country written in
# config.yaml is matched by its own name only.
KNOWN_COUNTRIES = (
    _CountryGroup(
        label="United States",
        names=("united states", "united states of america", "usa", "us", "u.s.",
               "u.s.a.", "america"),
        phrases=("united states", "usa", "u.s.a.", "u.s.", "silicon valley",
                 "santa clara", "sunnyvale", "san francisco", "new york",
                 "seattle", "austin"),
        tokens=("US",),
        state_codes=True,
    ),
    _CountryGroup(
        label="Israel",
        names=("israel",),
        phrases=("israel", "tel aviv", "haifa", "jerusalem", "herzliya", "raanana",
                 "ra'anana", "petah tikva", "yokneam", "netanya", "kfar saba",
                 "rehovot", "beer sheva", "hod hasharon", "caesarea"),
    ),
    _CountryGroup(
        label="United Arab Emirates",
        names=("united arab emirates", "uae", "u.a.e.", "emirates"),
        phrases=("united arab emirates", "uae", "u.a.e.", "dubai", "abu dhabi",
                 "sharjah", "ajman", "ras al khaimah"),
    ),
)

# US state codes that are not also used for a province, state or country elsewhere.
# Codes such as TN (Tamil Nadu), OR (Odisha), GA (Goa), MI (Milano), CT (Catania),
# FL (Firenze), IN (India), DE (Germany) or CA (Canada) are left out on purpose, so
# "Chennai, TN" or "Catania, CT" is never mistaken for a US job. US jobs that only
# carry one of those codes are caught by the LLM instead.
SAFE_STATE_CODES = set(
    "AK AZ DC HI IA KS KY MA MD NC ND NH NJ NM NV NY OH OK SC SD TX UT WA WI WV WY".split()
)

STATE_ENDING = re.compile(r",\s*([A-Z]{2})(?=\s*(?:\d{5}|,|;|\)|$))")
SEGMENT_SPLIT = re.compile(r"[;|\n]+")
HEADER_LOCATION = re.compile(r"^Location:\s*(.+)$", re.MULTILINE)


def _word_regex(items: tuple[str, ...], ignore_case: bool) -> re.Pattern | None:
    if not items:
        return None
    body = "|".join(re.escape(item) for item in sorted(items, key=len, reverse=True))
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", flags)


class _CountryMatcher:
    def __init__(self, label: str, phrases: tuple[str, ...],
                 tokens: tuple[str, ...], state_codes: bool):
        self.label = label
        self._phrases = _word_regex(phrases, ignore_case=True)
        self._tokens = _word_regex(tokens, ignore_case=False)
        self._state_codes = state_codes

    def find(self, text: str) -> str | None:
        """Return the text that matched, or None."""
        for regex in (self._phrases, self._tokens):
            if regex:
                match = regex.search(text)
                if match:
                    return match.group(0)
        if self._state_codes:
            for match in STATE_ENDING.finditer(text):
                if match.group(1) in SAFE_STATE_CODES:
                    return match.group(1)
        return None


def _build_matcher(configured_name: str) -> _CountryMatcher:
    key = configured_name.strip().lower()
    for group in KNOWN_COUNTRIES:
        if key in group.names:
            return _CountryMatcher(group.label, group.phrases, group.tokens,
                                   group.state_codes)
    # Unknown country: match its own name.
    return _CountryMatcher(configured_name.strip(), (configured_name.strip(),), (), False)


def location_from_page_text(page_text: str) -> str | None:
    """Read the 'Location:' header line that pagetext.py puts at the top of a posting."""
    match = HEADER_LOCATION.search(page_text)
    return match.group(1).strip() if match else None


class JobFilter:
    def __init__(self, blocked_countries: list[str], blocked_keywords: list[str]):
        self._matchers = [_build_matcher(name) for name in blocked_countries if name.strip()]
        self._keywords = [
            (kw.strip(), re.compile(rf"(?<!\w){re.escape(kw.strip())}(?!\w)", re.IGNORECASE))
            for kw in blocked_keywords
            if kw.strip()
        ]

    def check_location(self, location: str | None) -> FilterResult:
        """Block a job whose location is clearly in a blocked country.

        A location listing several places (separated by ; or |) is blocked only
        if every place is blocked, so a job open in Dublin and Austin passes.
        Works on a plain location ("Austin, TX") and on link text that has the
        location after the title ("Verification Engineer Austin, TX").
        """
        if not location or not self._matchers:
            return PASS

        segments = [s.strip() for s in SEGMENT_SPLIT.split(location) if s.strip()]
        if not segments:
            return PASS

        first_hit: tuple[str, str] | None = None
        for segment in segments:
            hit = self._first_hit(segment)
            if hit is None:
                return PASS  # at least one acceptable (or unknown) place
            first_hit = first_hit or hit

        label, matched = first_hit
        return FilterResult(True, f"blocked country: {label} (matched '{matched}')")

    def _first_hit(self, segment: str) -> tuple[str, str] | None:
        for matcher in self._matchers:
            matched = matcher.find(segment)
            if matched:
                return matcher.label, matched
        return None

    def check_keywords(self, text: str) -> FilterResult:
        """Block a job whose text contains a blocked keyword or phrase."""
        for keyword, regex in self._keywords:
            if regex.search(text):
                return FilterResult(True, f"blocked keyword: {keyword}")
        return PASS

    def check_posting(self, page_text: str) -> FilterResult:
        """Run both checks on a posting reduced by pagetext.py."""
        result = self.check_location(location_from_page_text(page_text))
        if result.blocked:
            return result
        return self.check_keywords(page_text)
