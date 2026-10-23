"""Semantic retrieval over book blurbs — the R in RAG.

"Something like The Martian" or "a cosy mystery" can't be answered by substring
search: the words a customer uses rarely appear in a title, author or genre.
Each book is embedded once from its title, author, genre and blurb; a request
is embedded when it arrives, and books are ranked by cosine similarity.

At catalog sizes a bookstore assistant deals with, this needs no vector
database and no numpy: unit vectors live in memory and ranking is one dot
product per book. Document embeddings are cached on disk keyed by model and
text, so a restart costs nothing and editing one blurb re-embeds one book.

The embedder is injected, like the chat client, so tests run offline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from booknow.catalog import Book, Catalog
from booknow.config import DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass
class OpenAIEmbedder:
    client: Any
    model: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = 256

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self.client.embeddings.create(model=self.model, input=batch)
            # The API returns an index per item; don't rely on response order.
            for item in sorted(response.data, key=lambda d: d.index):
                vectors.append(list(item.embedding))
        return vectors


def document_text(book: Book) -> str:
    """What gets embedded for a book. Changing this invalidates cached vectors."""
    return f"{book.title} by {book.author}. Genre: {book.genre}. {book.blurb}".strip()


def _unit(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else list(vector)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()


def _load_cache(path: Path | None) -> dict[str, list[float]]:
    if path is None or not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8"))["vectors"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Ignoring unreadable embedding cache %s: %s", path, exc)
        return {}


def _save_cache(path: Path | None, vectors: dict[str, list[float]]) -> None:
    if path is None:
        return
    payload = {"vectors": {k: [round(x, 6) for x in v] for k, v in vectors.items()}}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:  # a read-only disk costs a re-embed next start, nothing more
        logger.warning("Could not write embedding cache %s: %s", path, exc)


@dataclass(frozen=True)
class Match:
    book: Book
    score: float


@dataclass
class BookIndex:
    """Embeddings for every book in a catalog, built on first use."""

    catalog: Catalog
    embedder: Embedder
    cache_path: Path | None = None
    _vectors: dict[Book, list[float]] | None = field(default=None, init=False, repr=False)

    def _ensure_built(self) -> dict[Book, list[float]]:
        if self._vectors is not None:
            return self._vectors
        cache = _load_cache(self.cache_path)
        keys = {book: _cache_key(self.embedder.model, document_text(book)) for book in self.catalog}
        missing = [book for book, key in keys.items() if key not in cache]
        if missing:
            logger.info("Embedding %d of %d books", len(missing), len(keys))
            fresh = self.embedder.embed([document_text(book) for book in missing])
            for book, vector in zip(missing, fresh, strict=True):
                cache[keys[book]] = vector
            _save_cache(self.cache_path, cache)
        self._vectors = {book: _unit(cache[key]) for book, key in keys.items()}
        return self._vectors

    def search(
        self, query: str | None = None, like: Book | None = None, limit: int = 5
    ) -> list[Match]:
        """Rank books against a description, a book, or both (vectors are summed)."""
        vectors = self._ensure_built()
        parts: list[list[float]] = []
        if query:
            parts.append(_unit(self.embedder.embed([query])[0]))
        if like is not None:
            if like not in vectors:
                raise LookupError(f"{like.title!r} is not in the index")
            parts.append(vectors[like])
        if not parts:
            raise ValueError("search needs a query, a book, or both")
        target = _unit([sum(xs) for xs in zip(*parts, strict=True)])
        matches = [
            Match(book, _dot(target, vector)) for book, vector in vectors.items() if book != like
        ]
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[: max(limit, 0)]


def build_index(client: Any, catalog: Catalog, model: str, cache_path: str | None) -> BookIndex:
    return BookIndex(
        catalog=catalog,
        embedder=OpenAIEmbedder(client, model=model),
        cache_path=Path(cache_path).expanduser() if cache_path else None,
    )
