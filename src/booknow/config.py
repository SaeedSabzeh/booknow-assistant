"""Runtime configuration, loaded from the environment (never from source)."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODEL = "gpt-4o-mini"

# Values people commonly leave in place of a real key. Treat them as missing.
_PLACEHOLDER_KEYS = {"api", "your-key-here", "sk-xxx", "changeme", "todo"}


class MissingAPIKeyError(RuntimeError):
    """Raised when no usable OPENAI_API_KEY is available."""


def _load_dotenv_if_available() -> None:
    try:  # optional dependency; the app works fine without it
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    """Everything tunable in one place, so nothing is hard-coded in the logic."""

    api_key: str
    model: str = DEFAULT_MODEL
    temperature: float = 0.2
    max_tool_rounds: int = 4
    request_timeout: float = 30.0

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv_if_available()
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key or key.lower() in _PLACEHOLDER_KEYS:
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is not set.\n"
                "  cp .env.example .env   # then paste your key into .env\n"
                "  export OPENAI_API_KEY=sk-...   # or set it in your shell\n"
                "Never commit the key itself."
            )
        return cls(
            api_key=key,
            model=os.getenv("BOOKNOW_MODEL", DEFAULT_MODEL),
            temperature=float(os.getenv("BOOKNOW_TEMPERATURE", "0.2")),
            max_tool_rounds=int(os.getenv("BOOKNOW_MAX_TOOL_ROUNDS", "4")),
            request_timeout=float(os.getenv("BOOKNOW_TIMEOUT", "30")),
        )
