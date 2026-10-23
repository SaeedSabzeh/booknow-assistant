"""BookNow — a tool-calling bookstore assistant built on the OpenAI Chat Completions API."""

from booknow.agent import BookstoreAgent
from booknow.catalog import Book, Catalog, load_default_catalog
from booknow.config import MissingAPIKeyError, Settings
from booknow.retrieval import BookIndex, OpenAIEmbedder

__all__ = [
    "BookstoreAgent",
    "Book",
    "BookIndex",
    "OpenAIEmbedder",
    "Catalog",
    "load_default_catalog",
    "Settings",
    "MissingAPIKeyError",
]
__version__ = "0.3.0"
