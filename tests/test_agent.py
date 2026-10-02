import json
from pathlib import Path
from types import SimpleNamespace

from app.agents.graph import run_agent
from app.agents.llm import FallbackLLM
from app.agents.nodes import _build_sql, _generate_sql, agent_node, answer_node
from app.agents.prompts import SYSTEM_PROMPT
from app.tracing.tracer import TraceRecorder
from app.tracing.validator import validate_answer


def test_agent_uses_sql_for_db_question():
    result = run_agent("How many 3-bedroom houses are there?", "user_001", "conv_agent_test")
    assert "final_answer" in result
    assert result["validation_status"] == "pass"
    trace_path = Path("logs/traces") / f"{result['trace_id']}.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert len(trace["evidence"]) == 1
    assert trace["evidence"][0]["trace_id"] == result["trace_id"]
    assert trace["evidence"][0]["tool_call_id"] == trace["tool_calls"][0]["tool_call_id"]


def test_lowest_average_price_by_city_builds_ascending_grouped_query():
    sql = _build_sql("Which city has the lowest average house price?")

    assert "GROUP BY city" in sql
    assert "AVG(price)" in sql
    assert "ORDER BY avg_price ASC" in sql


def test_unsupported_fallback_question_is_not_guessed(monkeypatch):
    tracer = TraceRecorder(user_id="test-user", conversation_id="test-conv")
    state = {
        "messages": [{"role": "user", "content": "What is the median listing price in Seattle?"}],
        "user_id": "test-user",
        "conversation_id": "test-conv",
        "tool_results": [],
        "final_answer": "",
        "validation_status": "pass",
    }
    monkeypatch.setattr("app.agents.nodes.build_llm", FallbackLLM)

    agent_node(state, tracer)
    answer_node(state, tracer)

    assert not tracer.data["tool_calls"]
    assert state["validation_status"] == "unsupported"
    assert "couldn't safely translate" in state["final_answer"]
    assert tracer.data["validation"]["status"] == "unsupported"


def test_grounding_checks_numeric_values_in_grouped_rows():
    result = validate_answer(
        "Which city has the lowest average house price?",
        "The lowest average is $250,000.74.",
        [{"rows": [["Boston", "250000.738333"]]}],
    )

    assert result["status"] == "pass"


def test_grounding_accepts_numbers_inside_returned_addresses():
    rows = [
        ["7398 Oak St", "New York", "NY", 6, "1.8", 4504, 1961, "1612085.25"],
        ["5049 River St", "New York", "NY", 6, "1.1", 4969, 1987, "1607972.45"],
    ]
    answer = (
        "address: 7398 Oak St, city: New York, state: NY, bedrooms: 6, bathrooms: 1.8, "
        "area_sqft: 4504, year_built: 1961, price: 1612085.25\n"
        "address: 5049 River St, city: New York, state: NY, bedrooms: 6, bathrooms: 1.1, "
        "area_sqft: 4969, year_built: 1987, price: 1607972.45"
    )

    result = validate_answer(
        "Show me the top 2 most expensive houses with at least 4 bedrooms.",
        answer,
        [{"rows": rows}],
    )

    assert result["status"] == "pass"

    hallucinated = validate_answer(
        "Show me the top 2 most expensive houses with at least 4 bedrooms.",
        answer + " The hidden third home's price is 9999999.",
        [{"rows": rows}],
    )
    assert hallucinated["status"] == "failed"


def test_anthropic_prompt_cache_marks_only_static_system_prompt(monkeypatch):
    class FakeLLM:
        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(
                content="SELECT COUNT(*) AS count FROM house_prices",
                usage_metadata={
                    "input_tokens": 13,
                    "input_token_details": {
                        "cache_read": 0,
                        "cache_creation": 0,
                        "ephemeral_5m_input_tokens": 1300,
                        "ephemeral_1h_input_tokens": 0,
                    },
                },
            )

    llm = FakeLLM()
    monkeypatch.setattr(
        "app.agents.nodes.get_settings",
        lambda: SimpleNamespace(ANTHROPIC_PROMPT_CACHE=True),
    )

    sql, cache_usage = _generate_sql("How many houses are there?", llm)

    system_block = llm.messages[0]["content"][0]
    assert len(SYSTEM_PROMPT.split()) >= 750
    assert system_block["cache_control"] == {"type": "ephemeral"}
    assert llm.messages[1] == {"role": "user", "content": "How many houses are there?"}
    assert sql == "SELECT COUNT(*) AS count FROM house_prices;"
    assert cache_usage == {
        "cache_read_tokens": 0,
        "cache_creation_tokens": 1300,
        "input_tokens": 13,
    }


def test_anthropic_prompt_cache_can_be_disabled(monkeypatch):
    class FakeLLM:
        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(content="UNSUPPORTED", usage_metadata={})

    llm = FakeLLM()
    monkeypatch.setattr(
        "app.agents.nodes.get_settings",
        lambda: SimpleNamespace(ANTHROPIC_PROMPT_CACHE=False),
    )

    sql, _ = _generate_sql("Unsupported question", llm)

    assert sql == ""
    assert llm.messages[0]["content"] == SYSTEM_PROMPT


def test_sql_guard_accepts_public_qualified_house_prices(monkeypatch):
    class FakeLLM:
        def invoke(self, messages):
            return SimpleNamespace(
                content="SELECT COUNT(*) AS house_count FROM public.house_prices",
                usage_metadata={},
            )

    monkeypatch.setattr(
        "app.agents.nodes.get_settings",
        lambda: SimpleNamespace(ANTHROPIC_PROMPT_CACHE=True),
    )

    sql, _ = _generate_sql("How many houses are there?", FakeLLM())

    assert sql == "SELECT COUNT(*) AS house_count FROM public.house_prices;"
