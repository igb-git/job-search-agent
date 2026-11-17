"""The one interface shared by all LLM backends (Ollama, Claude, OpenAI).

A backend only has to implement `complete()`. Everything else, including
turning the reply into a JSON object and retrying once when the model returns
something that is not valid JSON, lives here and is shared.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

JSON_REMINDER = (
    "\n\nYour previous reply was not a valid JSON object. "
    "Reply again with ONLY the JSON object: no explanation, no code fences."
)

# Reasoning models such as Qwen3 may print their thinking before the answer.
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
CODE_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


class LLMError(Exception):
    """The model could not be reached, or did not return a usable answer."""


def parse_json_object(text: str) -> dict:
    """Extract a JSON object from a model reply. Raises ValueError if there is none.

    Handles the usual ways models wrap JSON: <think> blocks, ```json fences,
    and a sentence of chatter before or after the object.
    """
    cleaned = THINK_BLOCK.sub("", text).strip()
    cleaned = CODE_FENCE.sub("", cleaned).strip()

    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object found in the model reply")


class LLMClient(ABC):
    """Base class for LLM backends."""

    provider: str = "unknown"

    def __init__(self, model: str):
        self.model = model

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    @abstractmethod
    def complete(self, system: str, prompt: str) -> str:
        """Send one request and return the model's text reply.

        Implementations use temperature 0 (or the closest equivalent) and must
        raise LLMError for connection problems, authentication errors, missing
        models and empty replies.
        """

    def complete_json(self, system: str, prompt: str) -> dict:
        """Like complete(), but returns a parsed JSON object.

        If the first reply is not valid JSON, asks once more with a reminder.
        Raises LLMError if the second reply is not valid either.
        """
        reply = self.complete(system, prompt)
        try:
            return parse_json_object(reply)
        except ValueError:
            pass

        reply = self.complete(system, prompt + JSON_REMINDER)
        try:
            return parse_json_object(reply)
        except ValueError as exc:
            snippet = reply.strip().replace("\n", " ")[:200]
            raise LLMError(
                f"{self.name} did not return valid JSON after a retry. "
                f"Reply started with: {snippet!r}"
            ) from exc
