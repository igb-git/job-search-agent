"""Picks the LLM backend named in config.yaml."""

from __future__ import annotations

from ..config import LLMConfig
from .base import LLMClient, LLMError

__all__ = ["LLMClient", "LLMError", "create_client"]


def create_client(config: LLMConfig) -> LLMClient:
    """Build the client for config.provider ('ollama', 'claude' or 'openai')."""
    if config.provider == "ollama":
        from .ollama import DEFAULT_HOST, OllamaClient

        return OllamaClient(config.model, host=config.host or DEFAULT_HOST)

    if config.provider == "claude":
        try:
            from .claude import ClaudeClient
        except ImportError:
            raise LLMError(_missing_package("anthropic", "claude")) from None
        return ClaudeClient(
            config.model, api_key=config.api_key or "", workspace_id=config.workspace_id
        )

    if config.provider == "openai":
        try:
            from .openai import OpenAIClient
        except ImportError:
            raise LLMError(_missing_package("openai", "openai")) from None
        return OpenAIClient(config.model, api_key=config.api_key or "")

    raise LLMError(f"Unknown LLM provider '{config.provider}'.")


def _missing_package(package: str, extra: str) -> str:
    return (
        f"The '{package}' library is not installed. "
        f"Install it with: pip install -e \".[{extra}]\""
    )
