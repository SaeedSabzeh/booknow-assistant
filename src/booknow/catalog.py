"""The bookstore catalog: data, lookup, and fuzzy title resolution.

A model paraphrases: it will ask for "atomic habits", "1984 by Orwell", or
"the Sorcerer's Stone" for the same shelf item. Exact dictionary lookup misses
most of those, and a miss surfaces to the user as "we don't carry that" — the
worst possible failure for a shop. Lookup here is normalized and fuzzy, and a
genuine miss carries suggestions rather than a dead end.
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

_ARTICLES = ("the ", "a ", "an ")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Casefold, strip accents/punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _PUNCT.sub(" ", text.lower())
    return _WS.sub(" ", text).strip()


def _drop_article(text: str) -> str:
    for article in _ARTICLES:
        if text.startswith(article):
            return text[len(article):]
    return text


@dataclass(frozen=True)
class Book:
    title: str
    author: str
    price_cents: int
    genre: str
    year: int
    stock: int = 0

    @property
    def price(self) -> str:
        return f"${self.price_cents / 100:,.2f}"

    @property
    def in_stock(self) -> bool:
        return self.stock > 0

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "author": self.author,
            "price": self.price,
            "genre": self.genre,
            "year": self.year,
            "in_stock": self.in_stock,
            "stock": self.stock,
        }


class BookNotFoundError(LookupError):
    def __init__(self, query: str, suggestions: list[str]):
        self.query = query
        self.suggestions = suggestions
        super().__init__(f"No catalog match for {query!r}")


@dataclass
class Catalog:
    books: list[Book]
    _index: dict[str, Book] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for book in self.books:
            key = normalize(book.title)
            self._index[key] = book
            self._index.setdefault(_drop_article(key), book)

    # --- construction -----------------------------------------------------
    @classmethod
    def from_records(cls, records: Iterable[dict]) -> Catalog:
        return cls([Book(**record) for record in records])

    @classmethod
    def from_json(cls, path: str | Path) -> Catalog:
        return cls.from_records(json.loads(Path(path).read_text(encoding="utf-8")))

    # --- lookup -----------------------------------------------------------
    def __iter__(self) -> Iterator[Book]:
        return iter(self.books)

    def __len__(self) -> int:
        return len(self.books)

    def titles(self) -> list[str]:
        return [book.title for book in self.books]

    def find(self, query: str, cutoff: float = 0.72) -> Book:
        """Exact -> normalized -> substring -> fuzzy. Raises BookNotFoundError."""
        key = normalize(query)
        if not key:
            raise BookNotFoundError(query, [])

        if key in self._index:
            return self._index[key]
        bare = _drop_article(key)
        if bare in self._index:
            return self._index[bare]

        contains = [b for b in self.books if bare and bare in normalize(b.title)]
        if len(contains) == 1:
            return contains[0]

        match = difflib.get_close_matches(bare, list(self._index), n=1, cutoff=cutoff)
        if match:
            return self._index[match[0]]

        raise BookNotFoundError(query, self.suggest(query))

    def suggest(self, query: str, limit: int = 3) -> list[str]:
        scored = sorted(
            self.books,
            key=lambda b: difflib.SequenceMatcher(
                None, normalize(query), normalize(b.title)
            ).ratio(),
            reverse=True,
        )
        return [b.title for b in scored[:limit]]

    def search(self, query: str, limit: int = 5) -> list[Book]:
        """Substring search across title, author and genre."""
        key = normalize(query)
        if not key:
            return self.books[:limit]
        hits = [
            b
            for b in self.books
            if key in normalize(b.title)
            or key in normalize(b.author)
            or key in normalize(b.genre)
        ]
        if hits:
            return hits[:limit]
        close = difflib.get_close_matches(key, list(self._index), n=limit, cutoff=0.5)
        return [self._index[m] for m in close]

    def by_genre(self, genre: str) -> list[Book]:
        key = normalize(genre)
        return [b for b in self.books if normalize(b.genre) == key]

    def genres(self) -> list[str]:
        return sorted({b.genre for b in self.books})


def load_default_catalog() -> Catalog:
    """Load the bundled catalog shipped inside the package."""
    with resources.files("booknow.data").joinpath("books.json").open(
        "r", encoding="utf-8"
    ) as fh:
        return Catalog.from_records(json.load(fh))
