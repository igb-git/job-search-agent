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

    def __init__(
        self,
        model: str,
        api_key: str,
        timeout_seconds: float = 120,
        workspace_id: str | None = None,
    ):
        """workspace_id: only needed when the API key is not tied to a workspace and
        Anthropic asks for an anthropic-workspace-id header."""
        super().__init__(model)
        headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
        # The library already retries rate limits and server errors twice.
        self._client = anthropic.Anthropic(
            api_key=api_key, timeout=timeout_seconds, default_headers=headers
        )

    def close(self) -> None:
        self._client.close()

    def _create(self, system: str, prompt: str):
        # No temperature setting: recent versions of the library, and the newest
        # models, do not accept one. A plain request works with every model.
        return self._client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

    def complete(self, system: str, prompt: str) -> str:
        try:
            response = self._create(system, prompt)
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
            raise LLMError(f"Claude API error {exc.status_code}: {str(exc)[:600]}") from exc

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        if not text.strip():
            raise LLMError(f"{self.name} returned an empty reply.")
        return text
