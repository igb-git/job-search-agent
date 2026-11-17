"""SQLite storage: every job the agent has seen, its score, and whether it was notified.

A job moves through these statuses:

    new       -> URL found on a results page, not processed yet
    scored    -> fetched and scored by the LLM
    rejected  -> dropped by a filter (blocked country, blocked keyword, ...)
    error     -> download or scoring failed; retried up to MAX_ATTEMPTS times

Jobs are identified by their canonical URL, so the same posting found through
several keywords or sources is only processed once.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

MAX_ATTEMPTS = 3

# Query parameters that only track where a visitor came from. Kept narrow on
# purpose: some sites use parameters like ?id= to identify the job itself.
TRACKING_PARAMS = {"gclid", "fbclid", "gh_src", "trk", "trackingid", "mc_cid", "mc_eid"}

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    url           TEXT NOT NULL UNIQUE,
    source_name   TEXT NOT NULL,
    keyword       TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'new'
                  CHECK (status IN ('new', 'scored', 'rejected', 'error')),
    title         TEXT,
    company       TEXT,
    location      TEXT,
    country       TEXT,
    tier          TEXT,
    sponsorship   TEXT,
    description   TEXT,
    score         REAL,
    reason        TEXT,
    reject_reason TEXT,
    attempts      INTEGER NOT NULL DEFAULT 0,
    last_error    TEXT,
    first_seen    TEXT NOT NULL,
    processed_at  TEXT,
    notified_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);
CREATE INDEX IF NOT EXISTS idx_jobs_notify ON jobs (status, notified_at, score);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_url(url: str) -> str:
    """Normalise a job URL so the same posting always produces the same string.

    Lowercases scheme and host, drops the #fragment, removes tracking
    parameters (utm_*, gclid, ...), sorts the remaining parameters and strips a
    trailing slash from the path.
    """
    parts = urlsplit(url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_PARAMS
    ]
    query.sort()
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), "")
    )


@dataclass(frozen=True)
class Job:
    id: int
    url: str
    source_name: str
    keyword: str
    status: str
    title: str | None
    company: str | None
    location: str | None
    country: str | None
    tier: str | None
    sponsorship: str | None
    description: str | None
    score: float | None
    reason: str | None
    reject_reason: str | None
    attempts: int
    last_error: str | None
    first_seen: str
    processed_at: str | None
    notified_at: str | None


class Database:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(SCHEMA)
        self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ---- adding jobs ---------------------------------------------------

    def has_job(self, url: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM jobs WHERE url = ?", (canonical_url(url),)
        ).fetchone()
        return row is not None

    def add_job(
        self,
        url: str,
        source_name: str,
        keyword: str,
        title: str | None = None,
        reject_reason: str | None = None,
        description: str | None = None,
    ) -> bool:
        """Register a newly found job. Returns False if it was already known.

        title is a provisional title (the link text) that scoring later replaces.
        If reject_reason is given the job is stored as already rejected, which is
        how jobs dropped by the link-level filter are recorded without downloading
        them.
        """
        now = _now()
        status = "rejected" if reject_reason else "new"
        cursor = self._conn.execute(
            "INSERT OR IGNORE INTO jobs "
            "(url, source_name, keyword, title, status, reject_reason, description, "
            "first_seen, processed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (canonical_url(url), source_name, keyword, title, status, reject_reason,
             description, now, now if reject_reason else None),
        )
        self._conn.commit()
        return cursor.rowcount == 1

    # ---- processing ----------------------------------------------------

    def pending_jobs(self, limit: int) -> list[Job]:
        """Jobs waiting to be processed: new ones, plus failed ones with retries left."""
        rows = self._conn.execute(
            "SELECT * FROM jobs "
            "WHERE status = 'new' OR (status = 'error' AND attempts < ?) "
            "ORDER BY id LIMIT ?",
            (MAX_ATTEMPTS, limit),
        ).fetchall()
        return [Job(**dict(row)) for row in rows]

    def save_description(self, job_id: int, description: str) -> None:
        """Keep the downloaded text so a retry does not need to download it again."""
        self._conn.execute(
            "UPDATE jobs SET description = ? WHERE id = ?", (description, job_id)
        )
        self._conn.commit()

    def save_score(
        self,
        job_id: int,
        *,
        title: str | None,
        company: str | None,
        location: str | None,
        country: str | None,
        tier: str | None,
        score: float,
        reason: str,
    ) -> None:
        self._conn.execute(
            "UPDATE jobs SET status = 'scored', title = ?, company = ?, location = ?, "
            "country = ?, tier = ?, score = ?, reason = ?, "
            "last_error = NULL, processed_at = ? WHERE id = ?",
            (title, company, location, country, tier, score, reason,
             _now(), job_id),
        )
        self._conn.commit()

    def mark_rejected(
        self,
        job_id: int,
        reason: str,
        *,
        title: str | None = None,
        location: str | None = None,
        country: str | None = None,
    ) -> None:
        """Record a rejection with its reason. Fields left as None keep their value."""
        self._conn.execute(
            "UPDATE jobs SET status = 'rejected', reject_reason = ?, "
            "title = COALESCE(?, title), location = COALESCE(?, location), "
            "country = COALESCE(?, country), processed_at = ? WHERE id = ?",
            (reason, title, location, country, _now(), job_id),
        )
        self._conn.commit()

    def mark_error(self, job_id: int, message: str) -> None:
        """Record a failure. The job is retried until MAX_ATTEMPTS is reached."""
        self._conn.execute(
            "UPDATE jobs SET status = 'error', attempts = attempts + 1, "
            "last_error = ?, processed_at = ? WHERE id = ?",
            (message[:500], _now(), job_id),
        )
        self._conn.commit()

    # ---- notifying -----------------------------------------------------

    def unnotified_matches(self, threshold: float, limit: int) -> list[Job]:
        """Best scored jobs at or above the threshold that were not sent yet."""
        rows = self._conn.execute(
            "SELECT * FROM jobs "
            "WHERE status = 'scored' AND notified_at IS NULL AND score >= ? "
            "ORDER BY score DESC, id LIMIT ?",
            (threshold, limit),
        ).fetchall()
        return [Job(**dict(row)) for row in rows]

    def mark_notified(self, job_ids: list[int]) -> None:
        if not job_ids:
            return
        placeholders = ",".join("?" for _ in job_ids)
        self._conn.execute(
            f"UPDATE jobs SET notified_at = ? WHERE id IN ({placeholders})",
            [_now(), *job_ids],
        )
        self._conn.commit()

    # ---- reporting -----------------------------------------------------

    def status_counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
        ).fetchall()
        return {row["status"]: row["n"] for row in rows}
