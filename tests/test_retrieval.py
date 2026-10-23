import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from conftest import FakeEmbedder

from booknow.retrieval import BookIndex, OpenAIEmbedder, document_text


def titles(matches):
    return [m.book.title for m in matches]


def test_query_ranks_the_matching_blurb_first(index):
    assert titles(index.search("a cosy murder in a retirement village"))[0] == (
        "The Thursday Murder Club"
    )


def test_like_finds_the_nearest_neighbour_and_excludes_itself(index, shelf):
    martian = shelf.find("The Martian")
    results = titles(index.search(like=martian))
    assert results[0] == "Project Hail Mary"
    assert "The Martian" not in results


def test_query_and_like_combine(index, shelf):
    results = index.search("romance and marriage", like=shelf.find("The Martian"), limit=2)
    assert {m.book.title for m in results} == {"Project Hail Mary", "Pride and Prejudice"}


def test_scores_are_sorted_and_limit_applies(index):
    results = index.search("science", limit=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_search_needs_something_to_search_for(index):
    with pytest.raises(ValueError):
        index.search()


def test_index_is_built_lazily_and_once(index, embedder, shelf):
    assert embedder.calls == []
    index.search("space")
    index.search("murder")
    documents = [call for call in embedder.calls if len(call) == len(shelf)]
    assert len(documents) == 1
    assert len(embedder.calls) == 3  # one catalog batch, one per query


def test_disk_cache_spares_a_second_process_the_catalog_embedding(tmp_path, shelf):
    cache = tmp_path / "embeddings.json"
    BookIndex(shelf, FakeEmbedder(), cache_path=cache).search("space")
    assert cache.exists()

    fresh = FakeEmbedder()
    BookIndex(shelf, fresh, cache_path=cache).search("space")
    assert fresh.calls == [["space"]]


def test_editing_one_blurb_re_embeds_only_that_book(tmp_path, shelf):
    cache = tmp_path / "embeddings.json"
    BookIndex(shelf, FakeEmbedder(), cache_path=cache).search("space")

    records = [asdict(book) for book in shelf]
    records[2]["blurb"] = "Now a story about gardening."
    edited = type(shelf).from_records(records)
    fresh = FakeEmbedder()
    BookIndex(edited, fresh, cache_path=cache).search("space")
    assert fresh.calls[0] == [document_text(edited.books[2])]


def test_changing_the_model_invalidates_the_cache(tmp_path, shelf):
    cache = tmp_path / "embeddings.json"
    BookIndex(shelf, FakeEmbedder(), cache_path=cache).search("space")
    other = FakeEmbedder(model="another-model")
    BookIndex(shelf, other, cache_path=cache).search("space")
    assert len(other.calls[0]) == len(shelf)


def test_corrupt_cache_is_ignored(tmp_path, shelf):
    cache = tmp_path / "embeddings.json"
    cache.write_text("{not json")
    index = BookIndex(shelf, FakeEmbedder(), cache_path=cache)
    assert titles(index.search("murder"))[0] == "The Thursday Murder Club"
    assert "vectors" in json.loads(cache.read_text())


def test_openai_embedder_batches_and_restores_order():
    calls = []

    def create(model, input):
        calls.append(input)
        # Return items out of order: the embedder must sort by index.
        data = [SimpleNamespace(index=i, embedding=[float(len(t))]) for i, t in enumerate(input)]
        return SimpleNamespace(data=list(reversed(data)))

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    vectors = OpenAIEmbedder(client, model="m", batch_size=2).embed(["a", "bb", "ccc"])
    assert calls == [["a", "bb"], ["ccc"]]
    assert vectors == [[1.0], [2.0], [3.0]]
