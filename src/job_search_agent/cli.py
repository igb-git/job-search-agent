"""Command line interface.

    python -m job_search_agent                    one full run
    python -m job_search_agent --dry-run          run, but print instead of sending
    python -m job_search_agent --check-config     validate config.yaml and sources.txt
    python -m job_search_agent --check-sources    test each source and its link pattern
    python -m job_search_agent --test-telegram    send a test message
"""

from __future__ import annotations

import argparse
import logging
import sys

from .adapters import ADAPTERS, AdapterError
from .config import Config, ConfigError, load_config
from .fetcher import Fetcher, FetchError
from .links import extract_links, filter_by_pattern
from .llm.base import LLMError
from .notify.telegram import NotifyError, TelegramNotifier
from .pipeline import run
from .sources import SourcesError, build_search_targets, load_sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="job_search_agent",
        description="Finds job postings on company career pages, scores them "
        "against your profile and sends the best to Telegram.",
    )
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--env", default=".env", help="path to the .env file")
    parser.add_argument("--dry-run", action="store_true",
                        help="print matches instead of sending them to Telegram")
    parser.add_argument("--check-config", action="store_true",
                        help="validate the configuration and exit")
    parser.add_argument("--check-sources", action="store_true",
                        help="download the first keyword for each source and show the job links found")
    parser.add_argument("--test-telegram", action="store_true",
                        help="send a test message to Telegram and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="more detailed logging")
    return parser


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # httpx logs every request URL at INFO level. For Telegram that URL contains
    # the bot token, so these loggers must never be allowed to print it.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ---- sub-commands ----------------------------------------------------------------


def _check_config(cfg: Config) -> int:
    sources = load_sources(cfg.sources_file)
    targets = build_search_targets(sources, cfg.keywords)
    llm = cfg.llm
    where = f" at {llm.host}" if llm.host else ""

    print("Configuration is valid.")
    print(f"  keywords         : {', '.join(cfg.keywords)}")
    print(f"  LLM              : {llm.provider} / {llm.model}{where}")
    print(f"  blocked countries: {', '.join(cfg.blocked_countries) or 'none'}")
    print(f"  blocked keywords : {', '.join(cfg.blocked_keywords) or 'none'}")
    print(f"  Telegram         : "
          f"{'configured' if cfg.telegram.bot_token and cfg.telegram.chat_id else 'NOT configured'}")
    print(f"  database         : {cfg.database}")
    print(f"  sources          : {len(sources)} ({len(targets)} searches per run)")
    for source in sources:
        if source.adapter:
            note = f"   (API source: {source.adapter})"
        elif source.link_pattern:
            note = ""
        else:
            note = "   <-- no link pattern, will be skipped"
        print(f"    - {source.name}{note}")
    return 0


def _check_sources(cfg: Config) -> int:
    sources = load_sources(cfg.sources_file)
    targets = build_search_targets(sources, cfg.keywords[:1])
    with Fetcher(cfg.limits.delay_seconds, cfg.limits.request_timeout_seconds) as fetcher:
        for target in targets:
            print(f"\n== {target.source_name} | {target.keyword}")
            if target.link_pattern is None:
                print("   no link pattern set, the agent skips this source")
                continue
            try:
                page = fetcher.fetch(target.url)
            except FetchError as exc:
                print(f"   FAILED: {exc.message}")
                continue
            links = extract_links(page.text, page.final_url)
            jobs = filter_by_pattern(links, target.link_pattern)
            print(f"   {len(page.text)} bytes | {len(links)} links | {len(jobs)} job links")
            for link in jobs[:3]:
                print(f"     {link.text[:60]:60}  {link.url}")

        for source in (s for s in sources if s.adapter):
            keyword = cfg.keywords[0]
            print(f"\n== {source.name} | {keyword}  (API source: {source.adapter})")
            try:
                found = ADAPTERS[source.adapter](fetcher, keyword)
            except (FetchError, AdapterError) as exc:
                print(f"   FAILED: {getattr(exc, 'message', exc)}")
                continue
            print(f"   {len(found)} jobs")
            for job in found[:3]:
                print(f"     {job.title[:50]:50}  {job.location or '?'}")
                print(f"       {source.job_url.replace('{id}', job.id)}")
    return 0


def _test_telegram(cfg: Config) -> int:
    with TelegramNotifier(cfg.telegram.bot_token, cfg.telegram.chat_id) as bot:
        bot.send_alert("Test message: Telegram is set up correctly.")
    print("Test message sent.")
    return 0


# ---- entry point -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    try:
        needs_telegram = args.test_telegram or not (
            args.dry_run or args.check_config or args.check_sources
        )
        cfg = load_config(args.config, args.env, require_telegram=needs_telegram)

        if args.check_config:
            return _check_config(cfg)
        if args.check_sources:
            return _check_sources(cfg)
        if args.test_telegram:
            return _test_telegram(cfg)

        stats = run(cfg, dry_run=args.dry_run)
        logging.getLogger("job_search_agent").info("Done: %s", stats.summary())
        return 1 if stats.notify_failed else 0

    except (ConfigError, SourcesError, LLMError, NotifyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
