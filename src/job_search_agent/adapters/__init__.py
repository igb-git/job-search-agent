"""Adapters for companies whose job list comes from a JSON API.

To add a company: write a module with a `search(fetcher, keyword)` function that
returns a list of ApiJob, and register it below under the name used in sources.txt.
"""

from __future__ import annotations

from typing import Callable

from ..fetcher import Fetcher
from . import amd, mediatek, micron, realtek, tencent
from .base import AdapterError, ApiJob, job_text

Adapter = Callable[[Fetcher, str], list[ApiJob]]

ADAPTERS: dict[str, Adapter] = {
    "amd": amd.search,
    "mediatek": mediatek.search,
    "micron": micron.search,
    "realtek": realtek.search,
    "tencent": tencent.search,
}

__all__ = ["ADAPTERS", "Adapter", "AdapterError", "ApiJob", "job_text"]
