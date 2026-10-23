"""Fake OpenAI client and embedder so the whole suite runs offline, with no API key."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from booknow.catalog import Catalog
from booknow.retrieval import BookIndex


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


@dataclass
class FakeEmbedder:
    """Bag-of-words hashed into a fixed number of dimensions.

    Texts that share content words get similar vectors, which is enough to make
    ranking assertions meaningful while staying deterministic and offline.
    """

    model: str = "fake-embedding"
    dims: int = 512
    calls: list[list[str]] = field(default_factory=list)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        for word in re.findall(r"[a-z]{4,}", text.lower()):
            digest = hashlib.md5(word.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dims] += 1.0
        return vector


@pytest.fixture
def shelf() -> Catalog:
    """A small catalog with blurbs, for retrieval tests."""
    return Catalog.from_records(
        [
            {"title": "The Martian", "author": "Andy Weir", "price_cents": 1599,
             "genre": "Science Fiction", "year": 2011, "stock": 8,
             "blurb": "An astronaut stranded on Mars uses science and engineering to survive."},
            {"title": "Project Hail Mary", "author": "Andy Weir", "price_cents": 1999,
             "genre": "Science Fiction", "year": 2021, "stock": 0,
             "blurb": "A lone astronaut uses science to survive a desperate mission in space."},
            {"title": "The Thursday Murder Club", "author": "Richard Osman",
             "price_cents": 1599, "genre": "Mystery", "year": 2020, "stock": 12,
             "blurb": "Pensioners in a retirement village investigate a cosy village murder."},
            {"title": "Pride and Prejudice", "author": "Jane Austen", "price_cents": 1099,
             "genre": "Classic", "year": 1813, "stock": 8,
             "blurb": "A witty romance of manners, marriage and first impressions."},
        ]
    )


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def index(shelf, embedder) -> BookIndex:
    return BookIndex(shelf, embedder)
