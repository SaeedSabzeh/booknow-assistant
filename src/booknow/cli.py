"""Terminal chat client — useful for quick manual testing without Gradio."""

from __future__ import annotations

import argparse
import logging

from booknow.agent import BookstoreAgent, build_client
from booknow.catalog import load_default_catalog
from booknow.config import MissingAPIKeyError, Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="booknow", description="Chat with the BOOKNOW assistant.")
    parser.add_argument("-m", "--message", help="Send one message and exit.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log every tool call.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        settings = Settings.from_env()
    except MissingAPIKeyError as exc:
        print(exc)
        return 1

    agent = BookstoreAgent(
        client=build_client(settings), catalog=load_default_catalog(), settings=settings
    )

    if args.message:
        print(agent.reply(args.message))
        return 0

    print("BOOKNOW assistant. Ctrl-C or 'exit' to quit.\n")
    history: list[dict[str, str]] = []
    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if user.lower() in {"exit", "quit"}:
            return 0
        if not user:
            continue
        answer = agent.reply(user, history)
        print(f"bot > {answer}\n")
        history += [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ]


if __name__ == "__main__":
    raise SystemExit(main())
