"""LLM backend for Claude, through the official anthropic library.

Needs ANTHROPIC_API_KEY in .env. Install the library with:
    pip install -e ".[claude]"
"""

from __future__ import annotations

import anthropic

from .base import LLMClient, LLMError

MAX_TOKENS = 1024  # the JSON answer is short


class ClaudeClient(LLMClient):
    provider = "claude"

    def __init__(self, model: str, api_key: str, timeout_seconds: float = 120):
        super().__init__(model)
        # The library already retries rate limits and server errors twice.
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def _create(self, system: str, prompt: str, with_temperature: bool):
        params: dict = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if with_temperature:
            params["temperature"] = 0
        return self._client.messages.create(**params)

    def complete(self, system: str, prompt: str) -> str:
        try:
            try:
                response = self._create(system, prompt, with_temperature=True)
            except anthropic.BadRequestError as exc:
                # Some models do not accept a temperature setting. Retry without it.
                if "temperature" not in str(exc).lower():
                    raise
                response = self._create(system, prompt, with_temperature=False)
        except anthropic.AuthenticationError as exc:
            raise LLMError(
                "Claude rejected the API key. Check ANTHROPIC_API_KEY in .env."
            ) from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError(f"Claude denied access: {str(exc)[:200]}") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(
                f"Claude has no model named '{self.model}'. "
                "Check llm.claude.model in config.yaml."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise LLMError(
                "Claude rate limit reached (or the account is out of credit)."
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise LLMError("Claude did not answer in time.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Cannot reach the Claude API. Check your internet connection.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}: {str(exc)[:200]}") from exc

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        if not text.strip():
            raise LLMError(f"{self.name} returned an empty reply.")
        return text
