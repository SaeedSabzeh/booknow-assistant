from booknow.tools import REGISTRY, dispatch, tool_schemas


def test_every_tool_exposes_a_valid_schema():
    schemas = tool_schemas()
    assert len(schemas) == len(REGISTRY)
    for schema in schemas:
        fn = schema["function"]
        assert schema["type"] == "function"
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
        assert fn["parameters"]["additionalProperties"] is False


def test_get_price_hit(catalog):
    result = dispatch("get_price", {"title": "atomic habits"}, catalog)
    assert result["found"] is True
    assert result["price"] == "$18.75"


def test_get_price_miss_returns_suggestions(catalog):
    result = dispatch("get_price", {"title": "Dune"}, catalog)
    assert result["found"] is False
    assert result["did_you_mean"]


def test_unknown_tool_is_reported_not_raised(catalog):
    result = dispatch("order_pizza", {}, catalog)
    assert "error" in result
    assert "get_price" in result["available_tools"]


def test_bad_arguments_are_reported_not_raised(catalog):
    result = dispatch("get_price", {"wrong_kwarg": "x"}, catalog)
    assert "error" in result


def test_check_stock_out_of_stock_note(catalog):
    result = dispatch("check_stock", {"title": "The Silent Patient"}, catalog)
    assert result["in_stock"] is False
    assert result["note"]


def test_recommend_books_by_like_title(shelf, index):
    result = dispatch("recommend_books", {"like_title": "the martian"}, shelf, index=index)
    assert result["like_title"] == "The Martian"
    top = result["results"][0]
    assert top["title"] == "Project Hail Mary"
    assert top["blurb"] and "score" in top and top["in_stock"] is False


def test_recommend_books_unknown_title_returns_suggestions(shelf, index):
    result = dispatch("recommend_books", {"like_title": "Moby Dick"}, shelf, index=index)
    assert result["found"] is False
    assert result["did_you_mean"]


def test_recommend_books_needs_an_argument(shelf, index):
    assert "error" in dispatch("recommend_books", {}, shelf, index=index)


def test_recommend_books_without_an_index_is_reported_not_raised(shelf):
    assert "error" in dispatch("recommend_books", {"query": "space"}, shelf)


def test_index_tools_are_hidden_without_an_index():
    names = {s["function"]["name"] for s in tool_schemas(with_index=False)}
    assert "recommend_books" not in names
    assert "get_price" in names
