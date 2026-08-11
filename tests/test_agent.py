import json

from conftest import FakeClient, text_response, tool_response

from booknow.agent import BookstoreAgent, normalize_history


def make_agent(catalog, *scripted, **kwargs):
    client = FakeClient(scripted=list(scripted), **kwargs)
    return BookstoreAgent(client=client, catalog=catalog), client


def test_plain_answer_without_tools(catalog):
    agent, _ = make_agent(catalog, text_response("Hello!"))
    assert agent.reply("hi") == "Hello!"
    assert agent.tool_calls_made == []


def test_single_tool_call_round_trip(catalog):
    agent, client = make_agent(
        catalog,
        tool_response(("get_price", {"title": "atomic habits"})),
        text_response("Atomic Habits is $18.75."),
    )
    assert agent.reply("how much is atomic habits?") == "Atomic Habits is $18.75."
    tool_msgs = [m for m in client.calls[-1]["messages"] if m["role"] == "tool"]
    assert json.loads(tool_msgs[0]["content"])["price"] == "$18.75"
    assert tool_msgs[0]["tool_call_id"] == "call_0"


def test_handles_multiple_tool_calls_in_one_response(catalog):
    """One tool message per tool_call_id, or the model answers from thin air."""
    agent, client = make_agent(
        catalog,
        tool_response(
            ("get_price", {"title": "Atomic Habits"}),
            ("check_stock", {"title": "The Silent Patient"}),
        ),
        text_response("done"),
    )
    agent.reply("price of atomic habits and is the silent patient in stock?")
    assert agent.tool_calls_made == ["get_price", "check_stock"]
    tool_msgs = [m for m in client.calls[-1]["messages"] if m["role"] == "tool"]
    assert {m["tool_call_id"] for m in tool_msgs} == {"call_0", "call_1"}


def test_multi_round_tool_use(catalog):
    agent, _ = make_agent(
        catalog,
        tool_response(("search_books", {"query": "psychology"})),
        tool_response(("get_price", {"title": "Thinking, Fast and Slow"})),
        text_response("$22.50"),
    )
    assert agent.reply("cheapest psychology book?") == "$22.50"
    assert agent.tool_calls_made == ["search_books", "get_price"]


def test_tool_round_budget_forces_a_text_answer(catalog):
    agent, client = make_agent(
        catalog,
        *([tool_response(("get_price", {"title": "Atomic Habits"}))] * 4),
        text_response("Final answer."),
    )
    assert agent.reply("loop please") == "Final answer."
    assert "tools" not in client.calls[-1]


def test_malformed_tool_arguments_do_not_crash(catalog):
    from conftest import FakeChoice, FakeCompletion, FakeFunction, FakeMessage, FakeToolCall

    call = FakeToolCall("call_0", FakeFunction("get_price", "{not json"))
    broken = FakeCompletion([FakeChoice(FakeMessage(tool_calls=[call]))])
    agent, _ = make_agent(catalog, broken, text_response("recovered"))
    assert agent.reply("...") == "recovered"


def test_transient_api_errors_are_retried(catalog, monkeypatch):
    monkeypatch.setattr("booknow.agent.time.sleep", lambda _s: None)
    agent, _ = make_agent(catalog, text_response("ok"), raise_times=2)
    assert agent.reply("hi") == "ok"


def test_history_normalizes_tuples_and_dicts():
    assert normalize_history([("hi", "hello")]) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert normalize_history([{"role": "user", "content": "hi"}]) == [
        {"role": "user", "content": "hi"}
    ]
    assert normalize_history(None) == []
