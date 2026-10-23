"""Tool registry and dispatch.

Hand-written JSON schemas drift from the functions they describe, and an
if/elif dispatcher means adding a tool touches three places. A tool is
registered once here with `@tool(...)`; both the schema list sent to the API
and the dispatcher read from the registry, so they cannot disagree.

Dispatch never raises. A model calling a tool that does not exist, or passing
arguments that do not fit, is a normal event — it gets a structured error back
and recovers, rather than taking the request down.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from booknow.catalog import BookNotFoundError, Catalog

if TYPE_CHECKING:
    from booknow.retrieval import BookIndex

logger = logging.getLogger(__name__)

ToolFn = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn
    needs_index: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


REGISTRY: dict[str, Tool] = {}


def tool(
    name: str, description: str, parameters: dict[str, Any], needs_index: bool = False
) -> Callable[[ToolFn], ToolFn]:
    """Register a tool. With needs_index, it is called as fn(catalog, index, **args)."""

    def decorator(fn: ToolFn) -> ToolFn:
        REGISTRY[name] = Tool(
            name=name,
            description=description,
            parameters=parameters,
            fn=fn,
            needs_index=needs_index,
        )
        return fn

    return decorator


def _params(properties: dict[str, Any] | None = None, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
        "additionalProperties": False,
    }


# --- tools ----------------------------------------------------------------

@tool(
    name="get_price",
    description=(
        "Look up the price and stock of one book by title. Use whenever the user "
        "asks how much a book costs or whether it is available."
    ),
    parameters=_params(
        {"title": {"type": "string", "description": "Book title as the user said it."}},
        ["title"],
    ),
)
def get_price(catalog: Catalog, title: str) -> dict[str, Any]:
    try:
        book = catalog.find(title)
    except BookNotFoundError as exc:
        return {"found": False, "query": title, "did_you_mean": exc.suggestions}
    return {"found": True, "query": title, **book.to_dict()}


@tool(
    name="list_books",
    description="List the books the store carries. Optionally filter by genre.",
    parameters=_params(
        {
            "genre": {"type": "string", "description": "Optional genre filter."},
            "limit": {"type": "integer", "description": "Max results, default 20."},
        }
    ),
)
def list_books(catalog: Catalog, genre: str | None = None, limit: int = 20) -> dict[str, Any]:
    books = catalog.by_genre(genre) if genre else list(catalog)
    return {
        "count": len(books),
        "genres": catalog.genres(),
        "books": [b.to_dict() for b in books[:limit]],
    }


@tool(
    name="search_books",
    description=(
        "Free-text search over title, author and genre. Use for 'do you have "
        "anything by X' or 'something about habits'."
    ),
    parameters=_params(
        {
            "query": {"type": "string", "description": "Search terms."},
            "limit": {"type": "integer", "description": "Max results, default 5."},
        },
        ["query"],
    ),
)
def search_books(catalog: Catalog, query: str, limit: int = 5) -> dict[str, Any]:
    hits = catalog.search(query, limit=limit)
    return {"query": query, "count": len(hits), "results": [b.to_dict() for b in hits]}


@tool(
    name="check_stock",
    description="Check whether a specific title is currently in stock, and how many copies.",
    parameters=_params({"title": {"type": "string", "description": "Book title."}}, ["title"]),
)
def check_stock(catalog: Catalog, title: str) -> dict[str, Any]:
    try:
        book = catalog.find(title)
    except BookNotFoundError as exc:
        return {"found": False, "query": title, "did_you_mean": exc.suggestions}
    return {
        "found": True,
        "title": book.title,
        "in_stock": book.in_stock,
        "stock": book.stock,
        "note": "Out of stock — can be ordered in, ~5 working days." if not book.in_stock else "",
    }


@tool(
    name="recommend_books",
    description=(
        "Recommend books by meaning, using each book's description. Use for "
        "'something like X', moods, themes and plots: 'a cosy mystery', 'books "
        "about grief', 'funny science fiction'. Pass like_title for books similar "
        "to one the user names, query for what they describe, or both."
    ),
    parameters=_params(
        {
            "query": {"type": "string", "description": "What the user wants, in their words."},
            "like_title": {
                "type": "string",
                "description": "A book the user liked; results will resemble it.",
            },
            "limit": {"type": "integer", "description": "Max results, default 5."},
        }
    ),
    needs_index=True,
)
def recommend_books(
    catalog: Catalog,
    index: BookIndex,
    query: str | None = None,
    like_title: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    if not (query or like_title):
        return {"error": "Pass query, like_title, or both."}
    like = None
    if like_title:
        try:
            like = catalog.find(like_title)
        except BookNotFoundError as exc:
            return {"found": False, "query": like_title, "did_you_mean": exc.suggestions}
    matches = index.search(query=query, like=like, limit=limit)
    return {
        "query": query,
        "like_title": like.title if like else None,
        "results": [
            {**m.book.to_dict(), "blurb": m.book.blurb, "score": round(m.score, 3)}
            for m in matches
        ],
    }


# --- dispatch -------------------------------------------------------------

def tool_schemas(with_index: bool = True) -> list[dict[str, Any]]:
    """Schemas for the API. Without an index, tools that need one are not offered."""
    return [t.schema() for t in REGISTRY.values() if with_index or not t.needs_index]


def dispatch(
    name: str, arguments: dict[str, Any], catalog: Catalog, index: BookIndex | None = None
) -> dict[str, Any]:
    """Run a tool by name. Never raises: errors come back as data for the model."""
    entry = REGISTRY.get(name)
    if entry is None:
        logger.warning("Model called unknown tool %r", name)
        return {"error": f"Unknown tool {name!r}", "available_tools": sorted(REGISTRY)}
    try:
        if entry.needs_index:
            if index is None:
                return {"error": f"{name} is unavailable: no search index is configured."}
            return entry.fn(catalog, index, **arguments)
        return entry.fn(catalog, **arguments)
    except TypeError as exc:
        logger.warning("Bad arguments for %s: %s", name, exc)
        return {"error": f"Invalid arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - surface to the model, keep the chat alive
        logger.exception("Tool %s failed", name)
        return {"error": f"{type(exc).__name__}: {exc}"}
