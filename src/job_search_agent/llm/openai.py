"""LLM backend for OpenAI, through the official openai library.

Needs OPENAI_API_KEY in .env. Install the library with:
    pip install -e ".[openai]"
"""

from __future__ import annotations

import openai

from .base import LLMClient, LLMError

# Request settings that some models reject. If the API says one of them is not
# supported, it is removed and the request is repeated.
OPTIONAL_PARAMS = ("temperature", "response_format")


class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(self, model: str, api_key: str, timeout_seconds: float = 120):
        super().__init__(model)
        # The library already retries rate limits and server errors twice.
        self._client = openai.OpenAI(api_key=api_key, timeout=timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def complete(self, system: str, prompt: str) -> str:
        params: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            # JSON mode: the reply is guaranteed to be syntactically valid JSON.
            "response_format": {"type": "json_object"},
        }

        try:
            response = self._create_dropping_unsupported(params)
        except openai.AuthenticationError as exc:
            raise LLMError(
                "OpenAI rejected the API key. Check OPENAI_API_KEY in .env."
            ) from exc
        except openai.PermissionDeniedError as exc:
            raise LLMError(f"OpenAI denied access: {str(exc)[:200]}") from exc
        except openai.NotFoundError as exc:
            raise LLMError(
                f"OpenAI has no model named '{self.model}' available to this account. "
                "Check llm.openai.model in config.yaml."
            ) from exc
        except openai.RateLimitError as exc:
            raise LLMError(
                "OpenAI rate limit reached (or the account is out of credit)."
            ) from exc
        except openai.APITimeoutError as exc:
            raise LLMError("OpenAI did not answer in time.") from exc
        except openai.APIConnectionError as exc:
            raise LLMError("Cannot reach the OpenAI API. Check your internet connection.") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"OpenAI API error {exc.status_code}: {str(exc)[:200]}") from exc

        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise LLMError(f"{self.name} returned an empty reply.")
        return content

    def _create_dropping_unsupported(self, params: dict):
        """Send the request, removing a setting the model says it does not support."""
        while True:
            try:
                return self._client.chat.completions.create(**params)
            except openai.BadRequestError as exc:
                text = str(exc).lower()
                unsupported = next(
                    (name for name in OPTIONAL_PARAMS if name in params and name in text),
                    None,
                )
                if unsupported is None:
                    raise LLMError(f"OpenAI rejected the request: {str(exc)[:200]}") from exc
                params.pop(unsupported)
