"""BookNow — a tool-calling bookstore assistant built on the OpenAI Chat Completions API."""

from booknow.agent import BookstoreAgent
from booknow.catalog import Book, Catalog, load_default_catalog
from booknow.config import MissingAPIKeyError, Settings

__all__ = [
    "BookstoreAgent",
    "Book",
    "Catalog",
    "load_default_catalog",
    "Settings",
    "MissingAPIKeyError",
]
__version__ = "0.2.0"
