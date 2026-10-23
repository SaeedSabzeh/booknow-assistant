"""Gradio chat UI."""

from __future__ import annotations

import logging

import gradio as gr

from booknow.agent import BookstoreAgent, build_client
from booknow.catalog import load_default_catalog
from booknow.config import MissingAPIKeyError, Settings
from booknow.retrieval import build_index

logger = logging.getLogger(__name__)

EXAMPLES = [
    "How much is atomic habits?",
    "Do you have anything by Kahneman?",
    "Is The Silent Patient in stock, and what does it cost?",
    "What self-help books do you carry?",
    "I loved The Martian. What should I read next?",
    "Recommend a cosy mystery that's in stock.",
]


def build_demo() -> gr.ChatInterface:
    settings = Settings.from_env()
    client = build_client(settings)
    catalog = load_default_catalog()
    agent = BookstoreAgent(
        client=client,
        catalog=catalog,
        settings=settings,
        index=build_index(client, catalog, settings.embedding_model, settings.embedding_cache),
    )

    def respond(message: str, history: list[dict]) -> str:
        try:
            return agent.reply(message, history)
        except Exception:  # noqa: BLE001 - never crash the UI
            logger.exception("Reply failed")
            return "Sorry — something went wrong on my side. Please try again."

    return gr.ChatInterface(
        fn=respond,
        type="messages",
        title="BOOKNOW",
        description="Ask about prices, stock and what we carry, or for a recommendation.",
        examples=EXAMPLES,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        demo = build_demo()
    except MissingAPIKeyError as exc:
        raise SystemExit(str(exc)) from exc
    demo.launch()


if __name__ == "__main__":
    main()
