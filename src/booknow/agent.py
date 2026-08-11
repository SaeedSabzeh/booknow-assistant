"""The conversation loop.

Three things make this loop non-trivial:

  * A single response can contain several tool calls. The API requires one
    `tool` message per `tool_call_id`; answer only the first and the model
    writes a reply citing a result it never received.
  * Tool use is iterative — search, then price the winner — so the loop runs
    until the model stops calling tools, capped to bound cost.
  * The client is injected, so the whole loop is testable offline against
    scripted responses instead of a live, paid, non-deterministic API.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from booknow.catalog import Catalog, load_default_catalog
from booknow.config import Settings
from booknow.tools import dispatch, tool_schemas

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the assistant for BOOKNOW, an independent bookstore.

Rules:
- Answer questions about prices, stock and the catalog using the provided tools.
  Never invent a price, a stock level, or a book the store does not carry.
- If a tool reports the book was not found, say so plainly and offer the
  suggestions the tool returned.
- Prices are in USD and already formatted; repeat them exactly as given.
- Keep replies short and warm: two or three sentences unless asked for a list.
- If a question is unrelated to books or the store, say it's outside what you
  can help with and steer back to the catalog.
"""


class ChatClient(Protocol):
    """Structural type matching the bit of the OpenAI SDK we use."""

    chat: Any


def build_client(settings: Settings) -> ChatClient:
    from openai import OpenAI

    return OpenAI(api_key=settings.api_key, timeout=settings.request_timeout)


def _message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        payload = dict(message)
    elif hasattr(message, "model_dump"):
        payload = message.model_dump(exclude_none=True)
    else:  # pragma: no cover - defensive
        payload = {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", None),
        }
    payload.pop("function_call", None)
    payload.pop("annotations", None)
    return payload


def normalize_history(history: Sequence[Any] | None) -> list[dict[str, str]]:
    """Accept Gradio 'messages' dicts or legacy (user, bot) tuples."""
    normalized: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict) and item.get("content"):
            normalized.append({"role": item.get("role", "user"), "content": str(item["content"])})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, bot_msg = item
            if user_msg:
                normalized.append({"role": "user", "content": str(user_msg)})
            if bot_msg:
                normalized.append({"role": "assistant", "content": str(bot_msg)})
    return normalized


@dataclass
class BookstoreAgent:
    client: ChatClient
    catalog: Catalog = field(default_factory=load_default_catalog)
    settings: Settings | None = None
    system_prompt: str = SYSTEM_PROMPT
    max_api_retries: int = 3
    tool_calls_made: list[str] = field(default_factory=list, init=False)

    @property
    def model(self) -> str:
        return self.settings.model if self.settings else "gpt-4o-mini"

    @property
    def temperature(self) -> float:
        return self.settings.temperature if self.settings else 0.2

    @property
    def max_tool_rounds(self) -> int:
        return self.settings.max_tool_rounds if self.settings else 4

    # --- API plumbing -----------------------------------------------------
    def _create(self, messages: list[dict[str, Any]], with_tools: bool = True) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if with_tools:
            kwargs["tools"] = tool_schemas()
        last_error: Exception | None = None
        for attempt in range(self.max_api_retries):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - SDK raises many types
                last_error = exc
                if attempt == self.max_api_retries - 1:
                    break
                delay = 2**attempt + random.random()
                logger.warning("API call failed (%s); retrying in %.1fs", exc, delay)
                time.sleep(delay)
        raise RuntimeError(
            f"OpenAI request failed after {self.max_api_retries} attempts"
        ) from last_error

    def _run_tool_calls(self, message: Any) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for call in message.tool_calls or []:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
                logger.warning("Model sent unparseable arguments for %s", name)
            self.tool_calls_made.append(name)
            logger.info("tool=%s args=%s", name, arguments)
            payload = dispatch(name, arguments, self.catalog)
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": name,
                    "content": json.dumps(payload, ensure_ascii=False),
                }
            )
        return results

    # --- public API -------------------------------------------------------
    def reply(self, message: str, history: Sequence[Any] | None = None) -> str:
        self.tool_calls_made = []
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        messages += normalize_history(history)
        messages.append({"role": "user", "content": message})

        for _ in range(self.max_tool_rounds):
            response = self._create(messages)
            choice = response.choices[0]
            if not getattr(choice.message, "tool_calls", None):
                return choice.message.content or ""
            messages.append(_message_to_dict(choice.message))
            messages.extend(self._run_tool_calls(choice.message))

        # Tool budget exhausted: force a text-only answer from what we have.
        final = self._create(messages, with_tools=False)
        return final.choices[0].message.content or ""
