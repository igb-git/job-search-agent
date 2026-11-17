"""LLM backend for a local Ollama server.

Talks to Ollama's HTTP API directly with httpx, so no extra library is needed.
"""

from __future__ import annotations

import httpx

from .base import LLMClient, LLMError

DEFAULT_HOST = "http://localhost:11434"


class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        timeout_seconds: float = 300,
        num_ctx: int = 8192,
        json_mode: bool = True,
    ):
        """
        timeout_seconds: local models are slow, so this is much longer than the
            timeout used for downloading web pages.
        num_ctx: context window in tokens. Ollama's default is small and silently
            cuts long prompts, which would drop the end of a job description
            without any error. 8192 fits a job page plus your profile.
        json_mode: ask Ollama to force syntactically valid JSON output. The agent
            only makes JSON requests, so this stays on.
        """
        super().__init__(model)
        self._host = host.rstrip("/")
        self._num_ctx = num_ctx
        self._json_mode = json_mode
        self._client = httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def _post(self, payload: dict) -> httpx.Response:
        url = f"{self._host}/api/chat"
        try:
            return self._client.post(url, json=payload)
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self._host}. Is it running? "
                "(start it with 'ollama serve')"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"Ollama did not answer in time ({self.model}). "
                "The model may still be loading, or the machine is overloaded."
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

    def complete(self, system: str, prompt: str) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            # Turn off the long reasoning phase of models like Qwen3: it costs
            # minutes per job on a local machine and adds nothing to scoring.
            "think": False,
            "options": {"temperature": 0, "num_ctx": self._num_ctx},
        }
        if self._json_mode:
            payload["format"] = "json"

        response = self._post(payload)

        # Some models or older Ollama versions reject the "think" setting.
        # Retry once without it. Any <think> text is stripped by the JSON parser.
        if response.status_code == 400 and "think" in response.text.lower():
            payload.pop("think")
            response = self._post(payload)

        if response.status_code == 404:
            raise LLMError(
                f"Ollama has no model named '{self.model}'. "
                f"Check 'ollama list', or install it with 'ollama pull {self.model}'."
            )
        if response.status_code >= 400:
            raise LLMError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            content = response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError("Ollama returned a reply in an unexpected format.") from exc

        if not content or not content.strip():
            raise LLMError(f"{self.name} returned an empty reply.")
        return content
