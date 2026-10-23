# BOOKNOW — a tool-calling bookstore assistant

[![CI](https://github.com/SaeedSabzeh/booknow-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/SaeedSabzeh/booknow-assistant/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A small, production-shaped example of **LLM function calling** and **retrieval-augmented
generation**: a chat assistant for a fictional bookstore that answers questions about prices,
stock and the catalog by calling real Python functions instead of hallucinating answers, and
recommends books by meaning through semantic search over a 100-book catalog's blurbs.

Runs as a Gradio web app or in the terminal. The full test suite runs **offline, with no API
key**, because the model client is injected.

```
you > is the silent patient in stock and how much is atomic habits?
      ├─ check_stock(title="The Silent Patient")  -> {"in_stock": false, "stock": 0}
      └─ get_price(title="atomic habits")         -> {"price": "$18.75"}
bot > The Silent Patient is out of stock right now — we can order it in, about five
      working days. Atomic Habits is on the shelf at $18.75.

you > i loved the martian, what should i read next?
      └─ recommend_books(like_title="The Martian") -> Project Hail Mary, Dune, …
bot > Try Project Hail Mary, also by Andy Weir: another stranded astronaut solving
      their way out of trouble with science and humour. It's $19.99 and in stock.
```

## Quickstart

```bash
git clone https://github.com/SaeedSabzeh/booknow-assistant.git
cd booknow-assistant
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ui,dev]"

cp .env.example .env      # paste your OpenAI key into .env
make app                  # Gradio UI at http://127.0.0.1:7860
make run                  # or chat in the terminal (-v logs every tool call)
make test                 # 46 tests, no network, no key needed
```

## How it works

```
user turn
   │
   ▼
BookstoreAgent.reply()                  agent.py
   │  system prompt + history + message
   ▼
chat.completions.create(tools=…)        schemas from the registry
   │
   ├── no tool_calls ─────────────────► return text
   │
   └── tool_calls ──► dispatch(name, args, catalog)     tools.py
                          │
                          ▼
                      Catalog.find() — normalized + fuzzy lookup   catalog.py
                      BookIndex.search() — embeddings + cosine     retrieval.py
                          │                (recommend_books only)
                          ▼
                      append one tool message per call, loop again
                      (capped at BOOKNOW_MAX_TOOL_ROUNDS)
```

| Module | Responsibility |
| --- | --- |
| `catalog.py` | `Book` / `Catalog`, normalization, fuzzy title resolution, search |
| `tools.py` | `@tool` registry → JSON schemas + a dispatcher that never raises |
| `retrieval.py` | `BookIndex`: embeds blurbs, caches vectors on disk, ranks by cosine similarity |
| `agent.py` | conversation loop, multi-call and multi-round tool handling, retries |
| `app.py` / `cli.py` | Gradio UI and terminal REPL |
| `data/books.json` | the catalog — 100 books across 19 genres, each with a blurb |

### Adding a tool

One place, not three:

```python
@tool(
    name="cheapest_in_genre",
    description="Find the cheapest in-stock book in a genre.",
    parameters=_params({"genre": {"type": "string"}}, ["genre"]),
)
def cheapest_in_genre(catalog: Catalog, genre: str) -> dict:
    books = [b for b in catalog.by_genre(genre) if b.in_stock]
    return {"result": min(books, key=lambda b: b.price_cents).to_dict() if books else None}
```

A tool that needs the search index registers with `needs_index=True` and is called as
`fn(catalog, index, **args)`; see `recommend_books`.

The schema list sent to the API and the dispatcher both read from `REGISTRY`, so the new
tool is live immediately and `test_every_tool_exposes_a_valid_schema` covers it.

## Design notes

The interesting decisions, and what forced them.

**Fuzzy catalog lookup.** A model paraphrases the user, so `catalog["Atomic Habits"]`
misses "atomic habits", "Atomic Habits by James Clear", and every typo. A miss reaches
the customer as "we don't carry that", which is the one answer a shop must not give
wrongly. Lookup normalises, then falls back to substring and difflib matching; a real
miss returns candidates, so the model can ask instead of denying.

**Every tool call gets answered.** The API allows several tool calls in one response,
and requires exactly one `tool` message per `tool_call_id`. Answering only the first
leaves the model writing a confident sentence about a result it never received — a
hallucination created entirely by the plumbing. `test_handles_multiple_tool_calls_in_one_response`
guards this.

**Tool use loops.** "What's your cheapest psychology book?" is search, then price. A
single round can't do it. The loop runs to `max_tool_rounds`, then forces a text-only
answer so a confused model can't spend money in a circle.

**Dispatch never raises.** A bad tool name or malformed arguments are normal model
behaviour, not exceptional conditions. They come back as structured data the model can
recover from, instead of a 500 that ends the conversation.

**Registry over branching.** Hand-written schemas drift from the functions they
describe. `@tool(...)` registers once; the schema list and the dispatcher both derive
from it, and a schema-shape test covers every tool automatically.

**Retrieval is a tool, not a preamble.** The classic RAG pipeline retrieves on every
turn and pastes the hits into the prompt. Here most turns are "how much is X", where
retrieved blurbs are noise and cost tokens. `recommend_books` is just another tool: the
model calls it for "something like The Martian" or "a cosy mystery", and the results come
back with blurbs so it can say *why* a book fits. If no index is configured the tool isn't
offered at all, rather than offered and then failing.

**No vector database.** A hundred 1,536-dimension vectors fit in memory, and ranking is one
dot product per book, in pure Python with no numpy. A vector store would add a service to
run for no gain at this size; `BookIndex.search` is the seam to swap one in when the
catalog grows into the tens of thousands.

**Embeddings are cached by content.** The first recommendation embeds the catalog in one
batched call (roughly 8k tokens, a small fraction of a cent); vectors are written to a JSON cache keyed by a hash of model +
text. Restarts cost nothing, editing one blurb re-embeds one book, and switching the
embedding model invalidates everything automatically.

**Blurbs stay out of list payloads.** `Book.to_dict()` omits the blurb, so `list_books`
doesn't push 100 paragraphs into the context; only recommendations, which need the text
to justify a match, include it.

**Money as integer cents.** Formatted only at the boundary, in `Book.price`.

**The catalog is data.** `books.json`, not a dict in a module — a different shop is a
different file, not a fork.

**The client is injected, and so is the embedder.** Which is why 46 tests covering
multi-call turns, chained rounds, malformed arguments, retried failures, ranking and
cache invalidation run in well under a second for free.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | required |
| `BOOKNOW_MODEL` | `gpt-4o-mini` | any chat-completions model with tool support |
| `BOOKNOW_TEMPERATURE` | `0.2` | low, because answers should track the tool output |
| `BOOKNOW_MAX_TOOL_ROUNDS` | `4` | guard against tool loops |
| `BOOKNOW_TIMEOUT` | `30` | per-request timeout in seconds |
| `BOOKNOW_EMBEDDING_MODEL` | `text-embedding-3-small` | model used to embed blurbs and queries |
| `BOOKNOW_EMBEDDING_CACHE` | `~/.cache/booknow/embeddings.json` | vector cache; set empty to disable |

Swap the catalog without touching code:

```python
from booknow import BookstoreAgent, Catalog
agent = BookstoreAgent(client=client, catalog=Catalog.from_json("my_shop.json"))
```

Add a `"blurb"` to each record and pass an index to enable recommendations:

```python
from pathlib import Path
from booknow import BookIndex, OpenAIEmbedder
catalog = Catalog.from_json("my_shop.json")
index = BookIndex(catalog, OpenAIEmbedder(client), cache_path=Path("my_shop.vectors.json"))
agent = BookstoreAgent(client=client, catalog=catalog, index=index)
```

## Testing approach

`BookstoreAgent` takes its client by injection, so `tests/conftest.py` supplies a `FakeClient`
with a scripted sequence of responses. That makes the interesting paths — two tool calls in
one turn, chained rounds, malformed JSON arguments, retried API failures — cheap and
deterministic to assert on.

Retrieval works the same way: `FakeEmbedder` hashes words into a fixed-size vector, so texts
that share words land close together. That's enough to assert real ranking behaviour
("like The Martian" → Project Hail Mary) and cache behaviour (a second process embeds
nothing; an edited blurb re-embeds one book) without an API key.

```bash
pytest -q                       # offline
pytest --cov=booknow            # with coverage
```

## Roadmap

- [ ] Streamed responses in the Gradio UI (tool calls make this non-trivial)
- [ ] Swap hand-written schemas for Pydantic-generated ones
- [ ] Order placement tool with a confirmation step
- [x] RAG over book blurbs for "recommend me something like X"
- [ ] Token/cost accounting per conversation

## License

MIT — see [LICENSE](LICENSE).
