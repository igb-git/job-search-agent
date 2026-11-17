"""Sends job summaries and failure alerts to Telegram.

Uses the Bot API directly over HTTPS with httpx, so no Telegram library is
needed. Messages are formatted as HTML, which is safer than Markdown here:
job titles and company names often contain characters that break Markdown.
"""

from __future__ import annotations

import html
import time
from typing import Callable

import httpx

from ..db import Job

API_URL = "https://api.telegram.org/bot{token}/sendMessage"
MAX_MESSAGE_CHARS = 4000  # Telegram's hard limit is 4096; leave some room
HEADER_RESERVE = 80  # characters kept free for the header of each message
SEND_PAUSE_SECONDS = 1.0  # Telegram asks for at most about one message per second
MAX_RETRY_WAIT_SECONDS = 60


class NotifyError(Exception):
    """A message could not be sent. The text never contains the bot token."""


# ---- formatting (pure functions, no network) ---------------------------------


def format_job(job: Job) -> str:
    """One job as an HTML block."""
    title = html.escape(job.title or "(untitled)")
    url = html.escape(job.url, quote=True)
    score = f"{job.score:g}/10" if job.score is not None else "?/10"

    place = " · ".join(
        html.escape(part) for part in (job.company, job.location) if part
    )
    lines = [f'<b>{score}</b> · <a href="{url}">{title}</a>']
    if place:
        lines.append(place)
    if job.tier:
        lines.append(html.escape(job.tier))
    if job.reason:
        lines.append(f"<i>{html.escape(job.reason)}</i>")
    return "\n".join(lines)


def build_matches_messages(jobs: list[Job], max_chars: int = MAX_MESSAGE_CHARS) -> list[str]:
    """Group jobs into messages that fit Telegram's length limit.

    Every message starts with a header. When there is more than one message, the
    header also shows its position, for example (2/3).
    """
    if not jobs:
        return []

    room = max_chars - HEADER_RESERVE
    bodies: list[str] = []
    current: list[str] = []
    current_len = 0
    for job in jobs:
        block = format_job(job)[:room]  # a single job can never exceed the limit
        added = len(block) + 2  # the blank line between jobs
        if current and current_len + added > room:
            bodies.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(block)
        current_len += added
    bodies.append("\n\n".join(current))

    total = len(bodies)
    noun = "match" if len(jobs) == 1 else "matches"
    messages = []
    for number, body in enumerate(bodies, start=1):
        position = f" ({number}/{total})" if total > 1 else ""
        header = f"<b>{len(jobs)} new job {noun}</b>{position}"
        messages.append(f"{header}\n\n{body}")
    return messages


# ---- sending -----------------------------------------------------------------


class TelegramNotifier:
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        timeout_seconds: float = 30,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._url = API_URL.format(token=bot_token)
        self._chat_id = chat_id
        self._sleep = sleep
        self._client = httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "TelegramNotifier":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def send(self, text: str) -> None:
        """Send one HTML message. Raises NotifyError on failure."""
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }
        for attempt in (1, 2):
            try:
                response = self._client.post(self._url, json=payload)
            except httpx.HTTPError as exc:
                # Not str(exc): the message can contain the URL, which holds the token.
                if attempt == 1:
                    self._sleep(2)
                    continue
                raise NotifyError(
                    f"Could not reach Telegram ({type(exc).__name__})."
                ) from None

            if response.status_code == 200:
                return

            data = self._json(response)
            if response.status_code == 429 and attempt == 1:
                wait = data.get("parameters", {}).get("retry_after", 5)
                self._sleep(min(float(wait), MAX_RETRY_WAIT_SECONDS))
                continue
            raise NotifyError(self._explain(response.status_code, data))

    def send_all(self, messages: list[str]) -> None:
        """Send several messages in order, pausing between them."""
        for index, message in enumerate(messages):
            if index:
                self._sleep(SEND_PAUSE_SECONDS)
            self.send(message)

    def send_matches(self, jobs: list[Job]) -> int:
        """Send a summary of matching jobs. Returns the number of messages sent."""
        messages = build_matches_messages(jobs)
        self.send_all(messages)
        return len(messages)

    def send_alert(self, text: str) -> None:
        """Send a short failure or status notice."""
        self.send(f"⚠️ <b>job-search-agent</b>\n{html.escape(text)[:MAX_MESSAGE_CHARS - 100]}")

    @staticmethod
    def _json(response: httpx.Response) -> dict:
        try:
            data = response.json()
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _explain(status: int, data: dict) -> str:
        description = str(data.get("description", "")).strip()
        lowered = description.lower()
        if status in (401, 404):
            return "Telegram rejected the bot token. Check TELEGRAM_BOT_TOKEN in .env."
        if status == 400 and "chat not found" in lowered:
            return (
                "Telegram says the chat was not found. Check TELEGRAM_CHAT_ID, and make "
                "sure you have sent your bot a message first (a bot cannot start the chat)."
            )
        if status == 403:
            return f"Telegram refused the message: {description or 'the bot was blocked'}."
        if status == 429:
            return "Telegram rate limit hit again after waiting."
        return f"Telegram returned HTTP {status}: {description or 'no details'}."
