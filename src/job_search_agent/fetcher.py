"""Downloads web pages politely: a pause between requests to the same site,
retries on temporary failures, a browser-like user agent and a size limit.

Only plain HTTP downloads are supported. Pages that build their content with
JavaScript come back as an empty shell and are not handled here.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit

import httpx

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
MAX_ATTEMPTS = 3
MAX_RETRY_WAIT_SECONDS = 60
MAX_BYTES = 5 * 1024 * 1024


class FetchError(Exception):
    """A page could not be downloaded. The message says why."""

    def __init__(self, url: str, message: str):
        super().__init__(f"{url}: {message}")
        self.url = url
        self.message = message


@dataclass(frozen=True)
class FetchResult:
    url: str  # the URL that was requested
    final_url: str  # after redirects; use this to resolve relative links
    status_code: int
    text: str


class Fetcher:
    def __init__(
        self,
        delay_seconds: float = 3,
        timeout_seconds: float = 30,
        user_agent: str = DEFAULT_USER_AGENT,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._delay = delay_seconds
        self._sleep = sleep
        self._last_request: dict[str, float] = {}  # host -> time of last request
        self._client = httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Fetcher":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _wait_for_turn(self, host: str) -> None:
        """Pause so requests to the same host are spaced out, with some randomness."""
        last = self._last_request.get(host)
        if last is None:
            return
        wanted = self._delay * random.uniform(1.0, 1.5)
        elapsed = time.monotonic() - last
        if elapsed < wanted:
            self._sleep(wanted - elapsed)

    @staticmethod
    def _retry_wait(response: httpx.Response | None, attempt: int) -> float:
        """Seconds to wait before retrying. Honours a numeric Retry-After header."""
        if response is not None:
            header = response.headers.get("Retry-After", "")
            if header.isdigit():
                return min(float(header), MAX_RETRY_WAIT_SECONDS)
        return min(2.0**attempt, MAX_RETRY_WAIT_SECONDS)

    def fetch(self, url: str, headers: dict[str, str] | None = None) -> FetchResult:
        """Download a page, retrying temporary failures. Raises FetchError otherwise."""
        host = urlsplit(url).netloc.lower()
        last_problem = "unknown error"

        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._wait_for_turn(host)
            response: httpx.Response | None = None
            try:
                response = self._client.get(url, headers=headers)
            except httpx.UnsupportedProtocol as exc:
                raise FetchError(url, f"unsupported URL: {exc}") from exc
            except httpx.InvalidURL as exc:
                raise FetchError(url, f"invalid URL: {exc}") from exc
            except httpx.TransportError as exc:
                # Timeouts, connection errors, DNS failures: worth retrying.
                last_problem = f"{type(exc).__name__}: {exc}"
            except httpx.TooManyRedirects as exc:
                raise FetchError(
                    url,
                    "too many redirects (the site keeps redirecting this request; "
                    "it may need cookies or block automated downloads)",
                ) from exc
            except httpx.RequestError as exc:
                raise FetchError(url, f"request failed ({type(exc).__name__})") from exc
            finally:
                self._last_request[host] = time.monotonic()

            if response is not None:
                status = response.status_code
                if status == 429 or status >= 500:
                    last_problem = f"HTTP {status}"
                elif status >= 400:
                    # 403, 404 and friends will not fix themselves on retry.
                    raise FetchError(url, f"HTTP {status}")
                else:
                    return self._to_result(url, response)

            if attempt < MAX_ATTEMPTS:
                self._sleep(self._retry_wait(response, attempt))

        raise FetchError(url, f"gave up after {MAX_ATTEMPTS} attempts ({last_problem})")

    @staticmethod
    def _to_result(url: str, response: httpx.Response) -> FetchResult:
        content_type = response.headers.get("content-type", "").lower()
        if content_type and not any(
            kind in content_type for kind in ("html", "xml", "text/plain", "json")
        ):
            raise FetchError(url, f"not a web page (content-type: {content_type})")
        if len(response.content) > MAX_BYTES:
            raise FetchError(url, f"page is too large ({len(response.content)} bytes)")
        return FetchResult(
            url=url,
            final_url=str(response.url),
            status_code=response.status_code,
            text=response.text,
        )
