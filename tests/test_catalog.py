import pytest

from booknow.catalog import BookNotFoundError, load_default_catalog, normalize


def test_bundled_catalog_loads():
    catalog = load_default_catalog()
    assert len(catalog) == 20
    assert all(book.price.startswith("$") for book in catalog)


@pytest.mark.parametrize(
    "query",
    ["Atomic Habits", "atomic habits", "  ATOMIC   HABITS ", "atomic habbits", "Atomic Habits."],
)
def test_fuzzy_lookup_variants(catalog, query):
    """Every one of these is a phrasing a model actually produces."""
    assert catalog.find(query).title == "Atomic Habits"


def test_leading_article_is_optional(catalog):
    assert catalog.find("Silent Patient").title == "The Silent Patient"


def test_unknown_title_suggests_alternatives(catalog):
    with pytest.raises(BookNotFoundError) as excinfo:
        catalog.find("The Hobbit")
    assert excinfo.value.suggestions


def test_search_by_author_and_genre(catalog):
    assert catalog.search("kahneman")[0].title == "Thinking, Fast and Slow"
    assert {b.title for b in catalog.by_genre("thriller")} == {"The Silent Patient"}


def test_price_formatting(catalog):
    assert catalog.find("Atomic Habits").price == "$18.75"


def test_stock_flag(catalog):
    assert catalog.find("The Silent Patient").in_stock is False


def test_normalize_strips_smart_punctuation():
    expected = "harry potter and the sorcerer s stone"
    assert normalize("Harry Potter and the Sorcerer’s Stone") == expected
