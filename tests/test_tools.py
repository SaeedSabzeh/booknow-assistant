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
