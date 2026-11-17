"""Loads and validates config.yaml and .env.

Everything else in the project receives a `Config` object from `load_config()`
and never reads files or environment variables directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

VALID_PROVIDERS = ("ollama", "claude", "openai")
PLACEHOLDER_PREFIX = "put_your_"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class ConfigError(Exception):
    """Raised when the configuration is missing or invalid."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    host: str | None = None  # only used by ollama
    api_key: str | None = None  # only used by claude / openai


@dataclass(frozen=True)
class Limits:
    delay_seconds: float
    max_jobs_scored_per_run: int
    score_threshold: float
    max_matches_per_message: int
    request_timeout_seconds: float


@dataclass(frozen=True)
class Telegram:
    bot_token: str | None
    chat_id: str | None


@dataclass(frozen=True)
class Config:
    keywords: list[str]
    profile: str
    blocked_countries: list[str]
    blocked_keywords: list[str]
    llm: LLMConfig
    limits: Limits
    telegram: Telegram
    sources_file: Path
    database: Path


def _clean_secret(value: str | None) -> str | None:
    """Return None for empty values and for untouched placeholders."""
    if value is None:
        return None
    value = value.strip()
    if not value or value.startswith(PLACEHOLDER_PREFIX):
        return None
    return value


def _string_list(raw: object, name: str, errors: list[str], required: bool) -> list[str]:
    if raw is None:
        raw = []
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        errors.append(f"'{name}' must be a list of strings.")
        return []
    items = [x.strip() for x in raw if x.strip()]
    if required and not items:
        errors.append(f"'{name}' must contain at least one entry.")
    return items


def _number(
    section: dict, key: str, default: float, errors: list[str], minimum: float = 0
) -> float:
    value = section.get(key, default)
    # bool is a subclass of int, so exclude it explicitly
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"'limits.{key}' must be a number.")
        return default
    if value < minimum:
        errors.append(f"'limits.{key}' must be >= {minimum}.")
        return default
    return value


def _build_llm(raw: object, errors: list[str]) -> LLMConfig:
    fallback = LLMConfig(provider="ollama", model="")
    if not isinstance(raw, dict):
        errors.append("'llm' section is missing or is not a mapping.")
        return fallback

    provider = str(raw.get("provider", "")).strip().lower()
    if provider not in VALID_PROVIDERS:
        errors.append(
            f"'llm.provider' must be one of {', '.join(VALID_PROVIDERS)} "
            f"(got '{provider or 'nothing'}')."
        )
        return fallback

    section = raw.get(provider)
    if not isinstance(section, dict):
        errors.append(f"'llm.{provider}' section is missing.")
        return fallback

    model = str(section.get("model", "")).strip()
    if not model:
        errors.append(f"'llm.{provider}.model' is required.")

    if provider == "ollama":
        # OLLAMA_HOST in .env wins over config.yaml, then the default.
        host = (
            _clean_secret(os.environ.get("OLLAMA_HOST"))
            or str(section.get("host", "")).strip()
            or DEFAULT_OLLAMA_HOST
        )
        return LLMConfig(provider=provider, model=model, host=host)

    env_name = "ANTHROPIC_API_KEY" if provider == "claude" else "OPENAI_API_KEY"
    api_key = _clean_secret(os.environ.get(env_name))
    if api_key is None:
        errors.append(f"{env_name} is required in .env when llm.provider is '{provider}'.")
    return LLMConfig(provider=provider, model=model, api_key=api_key)


def load_config(
    config_path: str | Path = "config.yaml",
    env_path: str | Path = ".env",
    require_telegram: bool = True,
) -> Config:
    """Load and validate the configuration.

    Relative paths inside config.yaml (sources file, database) are resolved
    against the folder that contains config.yaml, so the agent works no matter
    which directory it is started from (important for cron).

    Set require_telegram=False for --dry-run, which sends nothing.
    """
    config_path = Path(config_path).resolve()
    env_path = Path(env_path)

    if not config_path.is_file():
        raise ConfigError(
            f"{config_path} not found. Copy config.example.yaml to config.yaml first."
        )

    # .env is optional (secrets can also come from real environment variables).
    # Real environment variables take precedence over the file.
    if env_path.is_file():
        load_dotenv(env_path, override=False)

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"config.yaml is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("config.yaml must contain a mapping at the top level.")

    errors: list[str] = []

    keywords = _string_list(data.get("keywords"), "keywords", errors, required=True)
    blocked_countries = _string_list(
        data.get("blocked_countries"), "blocked_countries", errors, required=False
    )
    blocked_keywords = _string_list(
        data.get("blocked_keywords"), "blocked_keywords", errors, required=False
    )

    profile = data.get("profile")
    if not isinstance(profile, str) or not profile.strip():
        errors.append("'profile' must be a non-empty text.")
        profile = ""

    llm = _build_llm(data.get("llm"), errors)

    raw_limits = data.get("limits") or {}
    if not isinstance(raw_limits, dict):
        errors.append("'limits' must be a mapping.")
        raw_limits = {}
    limits = Limits(
        delay_seconds=_number(raw_limits, "delay_seconds", 3, errors),
        max_jobs_scored_per_run=int(
            _number(raw_limits, "max_jobs_scored_per_run", 50, errors, minimum=1)
        ),
        score_threshold=_number(raw_limits, "score_threshold", 7, errors),
        max_matches_per_message=int(
            _number(raw_limits, "max_matches_per_message", 10, errors, minimum=1)
        ),
        request_timeout_seconds=_number(
            raw_limits, "request_timeout_seconds", 30, errors, minimum=1
        ),
    )

    telegram = Telegram(
        bot_token=_clean_secret(os.environ.get("TELEGRAM_BOT_TOKEN")),
        chat_id=_clean_secret(os.environ.get("TELEGRAM_CHAT_ID")),
    )
    if require_telegram:
        if telegram.bot_token is None:
            errors.append("TELEGRAM_BOT_TOKEN is missing in .env.")
        if telegram.chat_id is None:
            errors.append("TELEGRAM_CHAT_ID is missing in .env.")

    raw_paths = data.get("paths") or {}
    if not isinstance(raw_paths, dict):
        errors.append("'paths' must be a mapping.")
        raw_paths = {}
    base = config_path.parent
    sources_file = base / str(raw_paths.get("sources_file", "sources.txt"))
    database = base / str(raw_paths.get("database", "data/jobs.db"))

    if errors:
        bullet_list = "\n".join(f"  - {e}" for e in errors)
        raise ConfigError(f"Configuration problems:\n{bullet_list}")

    return Config(
        keywords=keywords,
        profile=profile.strip(),
        blocked_countries=blocked_countries,
        blocked_keywords=blocked_keywords,
        llm=llm,
        limits=limits,
        telegram=telegram,
        sources_file=sources_file,
        database=database,
    )
