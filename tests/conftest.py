"""Fake OpenAI client so the whole suite runs offline, with no API key."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from booknow.catalog import Catalog


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction
    type: str = "function"


@dataclass
class FakeMessage:
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[FakeToolCall] | None = None

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ]
        return {k: v for k, v in data.items() if not (exclude_none and v is None)}


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str = "stop"


@dataclass
class FakeCompletion:
    choices: list[FakeChoice]


class FakeCompletions:
    def __init__(self, owner: FakeClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> FakeCompletion:
        self._owner.calls.append(kwargs)
        if self._owner.raise_times > 0:
            self._owner.raise_times -= 1
            raise RuntimeError("transient upstream error")
        if not self._owner.scripted:
            return FakeCompletion([FakeChoice(FakeMessage(content="(no script left)"))])
        return self._owner.scripted.pop(0)


class FakeChat:
    def __init__(self, owner: FakeClient) -> None:
        self.completions = FakeCompletions(owner)


@dataclass
class FakeClient:
    scripted: list[FakeCompletion] = field(default_factory=list)
    raise_times: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.chat = FakeChat(self)


def text_response(text: str) -> FakeCompletion:
    return FakeCompletion([FakeChoice(FakeMessage(content=text))])


def tool_response(*calls: tuple[str, dict]) -> FakeCompletion:
    tool_calls = [
        FakeToolCall(id=f"call_{i}", function=FakeFunction(name=name, arguments=json.dumps(args)))
        for i, (name, args) in enumerate(calls)
    ]
    return FakeCompletion(
        [FakeChoice(FakeMessage(tool_calls=tool_calls), finish_reason="tool_calls")]
    )


@pytest.fixture
def catalog() -> Catalog:
    return Catalog.from_records(
        [
            {"title": "Atomic Habits", "author": "James Clear", "price_cents": 1875,
             "genre": "Self-help", "year": 2018, "stock": 25},
            {"title": "The Silent Patient", "author": "Alex Michaelides", "price_cents": 1299,
             "genre": "Thriller", "year": 2019, "stock": 0},
            {"title": "Thinking, Fast and Slow", "author": "Daniel Kahneman", "price_cents": 2250,
             "genre": "Psychology", "year": 2011, "stock": 13},
        ]
    )
