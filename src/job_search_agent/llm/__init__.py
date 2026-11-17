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
        from .claude import ClaudeClient

        return ClaudeClient(config.model, config.api_key)
 
    if config.provider == "openai":
        from .openai  import OpenAIClient

        return OpenAIClient(config.model, config.api_key) 

    raise LLMError(
        f"The '{config.provider}' backend is not available yet. "
        "Set llm.provider to 'ollama' in config.yaml for now."
    )
