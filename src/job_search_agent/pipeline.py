"""One full run of the agent: discover jobs, process new ones, notify.

    1. Discover  - download each search page, pick out job links, store new ones.
    2. Process   - download each new posting, filter it, score it with the LLM.
    3. Notify    - send the best unsent matches (and any problems) to Telegram.

Progress is stored in SQLite after every step, so an interrupted run loses
nothing and the next run carries on where this one stopped.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from dataclasses import dataclass
from urllib.parse import quote

from .adapters import ADAPTERS, AdapterError, job_text
from .config import Config
from .db import Database, Job
from .fetcher import Fetcher, FetchError
from .filters import JobFilter, location_from_page_text
from .links import extract_links, filter_by_pattern
from .llm import create_client
from .llm.base import LLMClient, LLMError
from .matcher import load_prompt_template, score_job
from .notify.telegram import NotifyError, TelegramNotifier
from .pagetext import extract_page_text
from .sources import SearchTarget, build_search_targets, load_sources

log = logging.getLogger(__name__)

# If the model fails this many jobs in a row, it is probably down: stop scoring
# instead of burning a retry attempt on every remaining job.
MAX_CONSECUTIVE_LLM_FAILURES = 3


@dataclass
class RunStats:
    searches: int = 0
    search_failures: int = 0
    new_jobs: int = 0
    prefiltered: int = 0  # rejected from the link text, never downloaded
    processed: int = 0
    rejected: int = 0
    scored: int = 0
    errors: int = 0
    matches_sent: int = 0
    notify_failed: bool = False

    def summary(self) -> str:
        return (
            f"searches {self.searches} ({self.search_failures} failed) | "
            f"new jobs {self.new_jobs} ({self.prefiltered} pre-filtered) | "
            f"processed {self.processed}: scored {self.scored}, "
            f"rejected {self.rejected}, errors {self.errors} | "
            f"matches {'shown' if self.matches_sent == 0 else 'sent'}: {self.matches_sent}"
        )


@dataclass
class _SourceReport:
    searched: int = 0
    failed: int = 0
    no_links: int = 0  # pages that downloaded but contained no links at all
    with_jobs: int = 0  # pages where at least one job link matched the pattern


# ---- step 1: discover ----------------------------------------------------------


def _discover(
    targets: list[SearchTarget],
    db: Database,
    fetcher: Fetcher,
    job_filter: JobFilter,
    stats: RunStats,
    problems: list[str],
) -> None:
    reports: dict[str, _SourceReport] = {}
    skipped: set[str] = set()

    for target in targets:
        if target.link_pattern is None:
            skipped.add(target.source_name)
            continue

        report = reports.setdefault(target.source_name, _SourceReport())
        report.searched += 1
        stats.searches += 1
        label = f"{target.source_name} | {target.keyword}"

        try:
            page = fetcher.fetch(target.url)
        except FetchError as exc:
            log.warning("[%s] download failed: %s", label, exc.message)
            report.failed += 1
            stats.search_failures += 1
            continue

        links = extract_links(page.text, page.final_url)
        if not links:
            report.no_links += 1
        job_links = filter_by_pattern(links, target.link_pattern)
        if job_links:
            report.with_jobs += 1

        new = 0
        for link in job_links:
            verdict = job_filter.check_location(link.text)
            added = db.add_job(
                link.url,
                target.source_name,
                target.keyword,
                title=link.text or None,
                reject_reason=verdict.reason if verdict.blocked else None,
            )
            if added:
                new += 1
                stats.new_jobs += 1
                if verdict.blocked:
                    stats.prefiltered += 1
        log.info("[%s] %d job links, %d new", label, len(job_links), new)

    for name in sorted(skipped):
        problems.append(
            f"{name}: no link pattern in sources.txt, so it was skipped. "
            "Add a third field with a pattern for its job URLs."
        )
    for name, report in reports.items():
        if report.with_jobs > 0:
            continue
        if report.failed == report.searched:
            problems.append(f"{name}: no search page could be downloaded.")
        elif report.no_links == report.searched - report.failed:
            problems.append(
                f"{name}: the pages contain no links at all, "
                "probably JavaScript-rendered. Park this source."
            )
        else:
            problems.append(
                f"{name}: no job links found for any keyword. The site layout or "
                "link pattern may have changed (or there are simply no openings)."
            )


def _discover_api(
    sources: list,
    keywords: list[str],
    db: Database,
    fetcher: Fetcher,
    job_filter: JobFilter,
    stats: RunStats,
    problems: list[str],
) -> None:
    """Query companies that expose a JSON API. Their jobs arrive with the full
    description, so the processing step does not need to download anything."""
    for source in sources:
        search = ADAPTERS[source.adapter]
        searched = failed = with_jobs = 0

        for keyword in keywords:
            searched += 1
            stats.searches += 1
            label = f"{source.name} | {keyword}"
            try:
                jobs = search(fetcher, keyword)
            except (FetchError, AdapterError) as exc:
                log.warning("[%s] query failed: %s", label, getattr(exc, "message", exc))
                failed += 1
                stats.search_failures += 1
                continue

            if jobs:
                with_jobs += 1
            new = 0
            for job in jobs:
                verdict = job_filter.check_location(job.location)
                added = db.add_job(
                    source.job_url.replace("{id}", quote(job.id, safe="")),
                    source.name,
                    keyword,
                    title=job.title,
                    reject_reason=verdict.reason if verdict.blocked else None,
                    description=None if verdict.blocked else job_text(job, source.name),
                )
                if added:
                    new += 1
                    stats.new_jobs += 1
                    if verdict.blocked:
                        stats.prefiltered += 1
            log.info("[%s] %d jobs, %d new", label, len(jobs), new)

        if with_jobs == 0:
            if failed == searched:
                problems.append(f"{source.name}: the API could not be queried for any keyword.")
            else:
                problems.append(
                    f"{source.name}: no jobs returned for any keyword (the API may have "
                    "changed, or there are simply no openings)."
                )


# ---- step 2: process -----------------------------------------------------------


def _process(
    cfg: Config,
    db: Database,
    fetcher: Fetcher,
    client: LLMClient,
    job_filter: JobFilter,
    stats: RunStats,
    problems: list[str],
) -> None:
    template = load_prompt_template()
    jobs = db.pending_jobs(cfg.limits.max_jobs_scored_per_run)
    log.info("%d jobs to process", len(jobs))
    llm_failures = 0

    for position, job in enumerate(jobs, start=1):
        label = f"[{position}/{len(jobs)}] {job.title or job.url}"
        stats.processed += 1

        # Download and reduce the posting, unless a previous attempt kept its text.
        text = job.description
        page_title = None
        if not text:
            try:
                page = extract_page_text(fetcher.fetch(job.url).text)
            except FetchError as exc:
                log.warning("%s: download failed: %s", label, exc.message)
                db.mark_error(job.id, exc.message)
                stats.errors += 1
                continue
            if page.looks_empty:
                log.warning("%s: page has almost no text (JavaScript?)", label)
                db.mark_error(job.id, "page has almost no text, probably JavaScript-rendered")
                stats.errors += 1
                continue
            text, page_title = page.text, page.title
            db.save_description(job.id, text)

        verdict = job_filter.check_posting(text)
        if verdict.blocked:
            log.info("%s: rejected (%s)", label, verdict.reason)
            db.mark_rejected(
                job.id, verdict.reason or "blocked",
                title=page_title, location=location_from_page_text(text),
            )
            stats.rejected += 1
            continue

        try:
            result = score_job(client, cfg.profile, cfg.blocked_countries, text, template)
        except LLMError as exc:
            log.warning("%s: scoring failed: %s", label, exc)
            db.mark_error(job.id, str(exc))
            stats.errors += 1
            llm_failures += 1
            if llm_failures >= MAX_CONSECUTIVE_LLM_FAILURES:
                problems.append(
                    f"Scoring stopped after {llm_failures} failures in a row: {exc}"
                )
                break
            continue
        llm_failures = 0

        # Second look at the country the model extracted.
        country_check = job_filter.check_location(result.country)
        if country_check.blocked:
            log.info("%s: rejected (%s, from the model)", label, country_check.reason)
            db.mark_rejected(
                job.id, f"{country_check.reason} (country reported by the model)",
                title=result.title, location=result.location, country=result.country,
            )
            stats.rejected += 1
            continue

        db.save_score(
            job.id,
            title=result.title or page_title or job.title,
            company=result.company,
            location=result.location,
            country=result.country,
            tier=result.tier,
            score=result.score,
            reason=result.reason,
        )
        stats.scored += 1
        log.info("%s: score %g (%s)", label, result.score, result.reason)


# ---- step 3: notify ------------------------------------------------------------


def _print_match(job: Job) -> None:
    print(f"\n{job.score:g}/10  {job.title or '(untitled)'}")
    print(f"  {job.company or '?'} | {job.location or '?'} | {job.tier or '?'}")
    print(f"  {job.reason}")
    print(f"  {job.url}")


def _notify(
    cfg: Config,
    db: Database,
    notifier: TelegramNotifier | None,
    problems: list[str],
    stats: RunStats,
) -> None:
    # Only the top matches go out each run. The rest stay unsent and are picked
    # up by the next run, so nothing is lost.
    matches = db.unnotified_matches(
        cfg.limits.score_threshold, cfg.limits.max_matches_per_message
    )

    if notifier is None:  # dry run: print instead of sending
        print(f"\n{len(matches)} matches at or above {cfg.limits.score_threshold:g}")
        for job in matches:
            _print_match(job)
        stats.matches_sent = 0
        if problems:
            print("\nProblems:")
            for problem in problems:
                print(f"  - {problem}")
        return

    if matches:
        try:
            notifier.send_matches(matches)
            db.mark_notified([job.id for job in matches])
            stats.matches_sent = len(matches)
        except NotifyError as exc:
            log.error("Could not send matches: %s", exc)
            stats.notify_failed = True
            return  # alerts would fail too, and the matches stay unsent

    if problems:
        try:
            notifier.send_alert("\n".join(f"• {p}" for p in problems))
        except NotifyError as exc:
            log.error("Could not send the alert: %s", exc)
            stats.notify_failed = True


# ---- the whole run -------------------------------------------------------------


def run(cfg: Config, dry_run: bool = False) -> RunStats:
    """Run one full cycle. With dry_run, nothing is sent to Telegram and nothing is
    marked as notified, but jobs are still downloaded, scored and stored, so a
    repeated run does not redo that work."""
    stats = RunStats()
    problems: list[str] = []

    sources = load_sources(cfg.sources_file)
    targets = build_search_targets(sources, cfg.keywords)
    job_filter = JobFilter(cfg.blocked_countries, cfg.blocked_keywords)

    with ExitStack() as stack:
        db = stack.enter_context(Database(cfg.database))
        fetcher = stack.enter_context(
            Fetcher(cfg.limits.delay_seconds, cfg.limits.request_timeout_seconds)
        )
        client = create_client(cfg.llm)
        if hasattr(client, "close"):
            stack.callback(client.close)

        notifier = None
        if not dry_run:
            notifier = stack.enter_context(
                TelegramNotifier(cfg.telegram.bot_token, cfg.telegram.chat_id)
            )

        log.info("Using %s", client.name)
        _discover(targets, db, fetcher, job_filter, stats, problems)
        _discover_api(
            [s for s in sources if s.adapter],
            cfg.keywords, db, fetcher, job_filter, stats, problems,
        )
        _process(cfg, db, fetcher, client, job_filter, stats, problems)
        _notify(cfg, db, notifier, problems, stats)

    return stats
