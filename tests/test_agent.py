import json
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
import time

from app.agents.graph import run_agent
from app.agents.llm import FallbackLLM
from app.agents.nodes import _build_sql, _generate_sql, _synthesize_answer_with_llm, agent_node, answer_node, execute_tools_node, planner_node
from app.agents.prompts import ANSWER_PROMPT, SYSTEM_PROMPT
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
    assert trace["tool_calls"][0]["decision_id"]
    assert any(event["event_type"] == "POLICY_CHECK" for event in trace["events"])
    assert any(event["event_type"] == "TOOL_OBSERVATION" for event in trace["events"])


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
    monkeypatch.setattr("app.agents.nodes.build_llm", lambda timeout_seconds=None: FallbackLLM())

    agent_node(state, tracer)
    answer_node(state, tracer)

    assert not tracer.data["tool_calls"]
    assert state["validation_status"] == "unsupported"
    assert "couldn't safely translate" in state["final_answer"]
    assert tracer.data["validation"]["status"] == "unsupported"


def test_non_database_question_does_not_call_model_or_tool(monkeypatch):
    tracer = TraceRecorder(user_id="test-user", conversation_id="test-conv")
    state = {
        "messages": [{"role": "user", "content": "Hello, what can you do?"}],
        "user_id": "test-user",
        "conversation_id": "test-conv",
        "short_term_context": [],
        "long_term_context": [],
        "tool_results": [],
        "observations": [],
        "started_at": time.monotonic(),
        "iteration": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    monkeypatch.setattr("app.agents.nodes.build_llm", lambda: (_ for _ in ()).throw(AssertionError("model must not be called")))

    planner_node(state, tracer)
    answer_node(state, tracer)

    assert state["validation_status"] == "not_applicable"
    assert tracer.data["llm_calls"] == []
    assert tracer.data["tool_calls"] == []
    assert any(event["event_type"] == "AGENT_DECISION" and event["decision"] == "no_tool" for event in tracer.data["events"])
    assert tracer.data["validation"]["status"] == "not_applicable"
    assert tracer.validate_contract() == []


def test_short_follow_up_uses_prior_database_context(monkeypatch):
    class FakeLLM:
        model = "fake-claude"

        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(content="SELECT COUNT(*) AS count FROM house_prices WHERE bedrooms = 3", usage_metadata={})

    fake_llm = FakeLLM()
    monkeypatch.setattr("app.agents.nodes.build_llm", lambda timeout_seconds=None: fake_llm)
    monkeypatch.setattr("app.agents.nodes.run_sql_tool", lambda *args: [{"tool": "sql", "columns": ["count"], "rows": [[17]], "row_count": 1, "evidence": {"trace_id": "trace-test", "tool_call_id": "tool-test"}}])
    tracer = TraceRecorder(user_id="test-user", conversation_id="test-conv")
    state = {
        "messages": [{"role": "user", "content": "What about those?"}],
        "user_id": "test-user",
        "conversation_id": "test-conv",
        "short_term_context": [{"role": "user", "content": "How many 3-bedroom houses are there?"}],
        "long_term_context": [],
        "tool_results": [],
        "observations": [],
        "started_at": time.monotonic(),
        "iteration": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
    }

    planner_node(state, tracer)

    assert state["planned_sql"]
    assert "Recent conversation" in fake_llm.messages[1]["content"]


def test_mixed_database_and_git_question_plans_independent_tools(monkeypatch):
    class FakeLLM:
        model = "fake-claude"

        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(content="SELECT COUNT(*) AS count FROM public.house_prices WHERE bedrooms = 3", usage_metadata={})

    fake_llm = FakeLLM()
    monkeypatch.setattr("app.agents.nodes.build_llm", lambda timeout_seconds=None: fake_llm)
    state = {
        "messages": [{"role": "user", "content": "How many 3-bedroom houses are there, and what is the Git repository status?"}],
        "user_id": "test-user",
        "conversation_id": "test-conv",
        "short_term_context": [],
        "long_term_context": [],
        "tool_results": [],
        "observations": [],
        "started_at": time.monotonic(),
        "iteration": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    tracer = TraceRecorder(user_id="test-user", conversation_id="test-conv")

    planner_node(state, tracer)

    assert state["planned_tools"] == ["sql", "git"]
    assert "Git repository status" not in fake_llm.messages[1]["content"]


def test_retryable_sql_error_schedules_bounded_retry(monkeypatch):
    tracer = TraceRecorder(user_id="test-user", conversation_id="test-conv")
    state = {
        "user_id": "test-user",
        "conversation_id": "test-conv",
        "planned_tools": ["sql"],
        "planned_sql": "SELECT missing FROM house_prices",
        "planned_decision_ids": {"sql": "decision-test"},
        "tool_results": [],
        "observations": [],
        "iteration": 1,
        "started_at": time.monotonic(),
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    monkeypatch.setattr(
        "app.agents.nodes.run_sql_tool",
        lambda *args: [{"tool": "sql", "error": {"type": "ProgrammingError", "message": "column does not exist"}}],
    )

    execute_tools_node(state, tracer)

    assert state["retry_requested"] is True
    assert state["observations"][-1]["status"] == "error"
    assert any(event["event_type"] == "AGENT_RETRY_SCHEDULED" for event in tracer.data["events"])


def test_independent_sql_and_git_tools_execute_in_parallel(monkeypatch):
    barrier = Barrier(2)

    def sql_tool(*args):
        barrier.wait(timeout=3)
        return [{"tool": "sql", "rows": [[3]], "row_count": 1}]

    def git_tool(*args):
        barrier.wait(timeout=3)
        return {"tool": "git", "tool_name": "git_status", "stdout": "## main", "returncode": 0}

    monkeypatch.setattr("app.agents.nodes.run_sql_tool", sql_tool)
    monkeypatch.setattr("app.agents.nodes.run_git_tool", git_tool)
    tracer = TraceRecorder()
    state = {
        "planned_tools": ["sql", "git"],
        "planned_sql": "SELECT COUNT(*) FROM house_prices",
        "planned_git_tool": "git_status",
        "planned_decision_ids": {"sql": "decision-sql", "git": "decision-git"},
        "user_id": "test-user",
        "conversation_id": "test-conv",
        "tool_results": [],
        "observations": [],
        "iteration": 1,
        "started_at": time.monotonic(),
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
    }

    execute_tools_node(state, tracer)

    assert {result["tool"] for result in state["tool_results"]} == {"sql", "git"}
    assert any(event["event_type"] == "TOOL_EXECUTION_PARALLEL_START" for event in tracer.data["events"])
    assert any(event["event_type"] == "TOOL_EXECUTION_PARALLEL_END" for event in tracer.data["events"])


def test_answer_synthesis_receives_tool_observation_and_evidence(monkeypatch):
    class FakeLLM:
        model = "fake-claude"

        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(content="There are 17 matching houses.", usage_metadata={"input_tokens": 30, "output_tokens": 8})

    fake_llm = FakeLLM()
    monkeypatch.setattr("app.agents.nodes.build_llm", lambda timeout_seconds=None: fake_llm)
    monkeypatch.setattr(
        "app.agents.nodes.get_settings",
        lambda: SimpleNamespace(
            AGENT_TIMEOUT_SECONDS=45.0,
            AGENT_MAX_TOKENS=12000,
            AGENT_MAX_COST_USD=0.05,
            LLM_INPUT_COST_PER_MILLION_USD=3.0,
            LLM_OUTPUT_COST_PER_MILLION_USD=15.0,
        ),
    )
    tracer = TraceRecorder()
    state = {"started_at": time.monotonic(), "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0}
    observations = [{"tool": "sql.execute_sql", "tool_call_id": "tool-a", "evidence_id": "ev-a", "columns": ["count"], "rows": [[17]]}]

    answer = _synthesize_answer_with_llm("How many houses?", "", observations, state, tracer)

    assert answer == "There are 17 matching houses."
    assert '"evidence_id": "ev-a"' in fake_llm.messages[1]["content"]
    assert tracer.data["llm_calls"][0]["purpose"] == "answer_synthesis"
    assert tracer.data["llm_calls"][0]["status"] == "success"


def test_answer_repair_prompt_and_usage_are_traced_and_budgeted(monkeypatch):
    class FakeLLM:
        model = "fake-claude"

        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(
                content="2542 Lake St has 6 bedrooms.",
                usage_metadata={"input_tokens": 100, "output_tokens": 20},
            )

    fake_llm = FakeLLM()
    monkeypatch.setattr("app.agents.nodes.build_llm", lambda timeout_seconds=None: fake_llm)
    monkeypatch.setattr(
        "app.agents.nodes.get_settings",
        lambda: SimpleNamespace(
            AGENT_TIMEOUT_SECONDS=45.0,
            AGENT_MAX_TOKENS=12000,
            AGENT_MAX_COST_USD=0.05,
            LLM_INPUT_COST_PER_MILLION_USD=3.0,
            LLM_OUTPUT_COST_PER_MILLION_USD=15.0,
        ),
    )
    state = {
        "started_at": time.monotonic(),
        "input_tokens": 10,
        "output_tokens": 5,
        "estimated_cost_usd": 0.001,
    }
    tracer = TraceRecorder()

    answer = _synthesize_answer_with_llm(
        "Show a house.",
        "",
        [{"tool": "sql.execute_sql", "rows": [["2542 Lake St", 6]]}],
        state,
        tracer,
        repair_feedback="Remove unsupported number 7.",
    )

    assert answer == "2542 Lake St has 6 bedrooms."
    assert '"repair_feedback": "Remove unsupported number 7."' in fake_llm.messages[1]["content"]
    assert state["input_tokens"] == 110
    assert state["output_tokens"] == 25
    assert abs(state["estimated_cost_usd"] - 0.0016) < 1e-12
    assert tracer.data["llm_calls"][0]["purpose"] == "answer_repair"
    assert "do not claim a home is suitable" in ANSWER_PROMPT


def test_grounding_checks_numeric_values_in_grouped_rows():
    result = validate_answer(
        "Which city has the lowest average house price?",
        "The lowest average is $250,000.74.",
        [{"tool": "sql", "columns": ["city", "avg_price"], "rows": [["Boston", "250000.738333"]], "evidence": {"evidence_id": "ev-test"}}],
    )

    assert result["status"] == "pass"
    assert result["claims"][0]["status"] == "pass"
    assert result["claims"][0]["evidence_ids"] == ["ev-test"]


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
        [{"tool": "sql", "columns": ["address", "city", "state", "bedrooms", "bathrooms", "area_sqft", "year_built", "price"], "rows": rows, "evidence": {"evidence_id": "ev-test"}}],
    )

    assert result["status"] == "pass"

    hallucinated = validate_answer(
        "Show me the top 2 most expensive houses with at least 4 bedrooms.",
        answer + " The hidden third home's price is 9999999.",
        [{"tool": "sql", "columns": ["address", "city", "state", "bedrooms", "bathrooms", "area_sqft", "year_built", "price"], "rows": rows, "evidence": {"evidence_id": "ev-test"}}],
    )
    assert hallucinated["status"] == "failed"


def test_unvalidated_ranked_house_answer_falls_back_to_grounded_rows(monkeypatch):
    question = "I want to know top 5 cheapest houses with at least 3 bedrooms then choose the best option for me ?"
    rows = [
        ["2542 Lake St", "Atlanta", "GA", 6, "2.7", 579, 1964, "295620.06"],
        ["7419 Pine St", "Atlanta", "GA", 3, "4.7", 555, 1983, "303661.18"],
        ["5139 Hill St", "Denver", "CO", 3, "4.3", 1084, 1957, "337977.91"],
        ["3971 Oak St", "Dallas", "TX", 3, "2.2", 593, 1983, "343619.04"],
        ["2549 Lake St", "Atlanta", "GA", 5, "1.0", 1149, 1952, "351728.66"],
    ]
    tracer = TraceRecorder(user_id="test-user", conversation_id="test-conv")
    tool_call_id = "tool-ranked-houses"
    sql_result = {
        "tool": "sql",
        "columns": ["address", "city", "state", "bedrooms", "bathrooms", "area_sqft", "year_built", "price"],
        "rows": rows,
        "tool_call_id": tool_call_id,
        "evidence": {"trace_id": tracer.trace_id, "tool_call_id": tool_call_id, "evidence_id": "ev-ranked-houses"},
    }
    state = {
        "messages": [{"role": "user", "content": question}],
        "tool_results": [sql_result],
        "final_answer": "",
        "validation_status": "pending",
    }
    monkeypatch.setattr(
        "app.agents.nodes._synthesize_answer_with_llm",
        lambda *args, **kwargs: "Best price per sqft is $312/sqft versus $511/sqft; choose the Denver home as best value.",
    )

    answer_node(state, tracer)

    assert state["validation_status"] == "pass"
    assert "Based on the lowest listed price" in state["final_answer"]
    assert "the recommended option is 2542 Lake St, Atlanta." in state["final_answer"]
    assert "312/sqft" not in state["final_answer"]
    assert any(event["event_type"] == "ANSWER_SYNTHESIS_FALLBACK" for event in tracer.data["events"])


def test_best_option_fallback_uses_grounded_room_counts_for_family_example():
    from app.agents.nodes import _format_grounded_sql_answer

    question = "I want to know top 3 highest houses in New York and choose the best option for my family of 5."
    sql_result = {
        "tool": "sql",
        "columns": ["address", "city", "price", "bedrooms", "bathrooms", "area_sqft", "year_built"],
        "rows": [
            ["7398 Oak St", "New York", "1612085.25", 6, "1.8", 4504, 1961],
            ["5049 River St", "New York", "1607972.45", 6, "1.1", 4969, 1987],
            ["7481 Maple St", "New York", "1549794.80", 3, "3.4", 4607, 1969],
        ],
        "evidence": {"evidence_id": "ev-family-houses"},
    }

    answer = _format_grounded_sql_answer(question, sql_result)
    validation = validate_answer(question, answer, [sql_result])

    assert validation["status"] == "pass"
    assert "the recommended option is 7398 Oak St, New York." in answer
    assert "Its database row lists 6 bedrooms and 1.8 bathrooms." in answer
    assert "family of 5" not in answer


def test_invalid_synthesis_is_repaired_once_before_template_fallback(monkeypatch):
    question = "Show the cheapest house and choose the best option."
    tracer = TraceRecorder(user_id="test-user", conversation_id="test-conv")
    tool_call_id = "tool-repair-test"
    sql_result = {
        "tool": "sql",
        "columns": ["address", "city", "price", "bedrooms", "bathrooms"],
        "rows": [["2542 Lake St", "Atlanta", "295620.06", 6, "2.7"]],
        "row_count": 1,
        "sql": "SELECT address, city, price, bedrooms, bathrooms FROM public.house_prices ORDER BY price ASC LIMIT 1",
        "tool_call_id": tool_call_id,
        "evidence": {"trace_id": tracer.trace_id, "tool_call_id": tool_call_id, "evidence_id": "ev-repair-test", "row_count": 1},
    }
    state = {
        "messages": [{"role": "user", "content": question}],
        "short_term_context": [],
        "long_term_context": [],
        "observations": [],
        "tool_results": [sql_result],
        "final_answer": "",
        "validation_status": "pending",
    }
    calls = []

    def synthesize(*args, repair_feedback=""):
        calls.append(repair_feedback)
        return "The house has 7 bedrooms." if not repair_feedback else "2542 Lake St has 6 bedrooms and 2.7 bathrooms."

    monkeypatch.setattr("app.agents.nodes._synthesize_answer_with_llm", synthesize)

    answer_node(state, tracer)

    assert len(calls) == 2
    assert calls[0] == ""
    assert "7" in calls[1]
    assert state["final_answer"] == "2542 Lake St has 6 bedrooms and 2.7 bathrooms."
    assert state["validation_status"] == "pass"
    assert [event["event_type"] for event in tracer.data["events"] if event["event_type"].startswith("ANSWER_REPAIR_")] == [
        "ANSWER_REPAIR_START", "ANSWER_REPAIR_END"
    ]
    assert next(event for event in tracer.data["events"] if event["event_type"] == "ANSWER_REPAIR_END")["status"] == "pass"


def test_failed_answer_repair_runs_once_then_uses_template(monkeypatch):
    question = "Show the cheapest house and choose the best option."
    tracer = TraceRecorder(user_id="test-user", conversation_id="test-conv")
    tool_call_id = "tool-repair-fallback-test"
    sql_result = {
        "tool": "sql",
        "columns": ["address", "city", "price", "bedrooms", "bathrooms"],
        "rows": [["2542 Lake St", "Atlanta", "295620.06", 6, "2.7"]],
        "row_count": 1,
        "sql": "SELECT address, city, price, bedrooms, bathrooms FROM public.house_prices ORDER BY price ASC LIMIT 1",
        "tool_call_id": tool_call_id,
        "evidence": {"trace_id": tracer.trace_id, "tool_call_id": tool_call_id, "evidence_id": "ev-repair-fallback", "row_count": 1},
    }
    state = {
        "messages": [{"role": "user", "content": question}],
        "short_term_context": [],
        "long_term_context": [],
        "observations": [],
        "tool_results": [sql_result],
        "final_answer": "",
        "validation_status": "pending",
    }
    calls = []
    monkeypatch.setattr(
        "app.agents.nodes._synthesize_answer_with_llm",
        lambda *args, **kwargs: calls.append(kwargs.get("repair_feedback", "")) or "There are 7 bedrooms.",
    )

    answer_node(state, tracer)

    assert len(calls) == 2
    assert state["validation_status"] == "pass"
    assert state["final_answer"].startswith("Matching results:")
    assert next(event for event in tracer.data["events"] if event["event_type"] == "ANSWER_REPAIR_END")["status"] == "failed"


def test_claim_values_cannot_be_borrowed_across_different_result_rows():
    result = validate_answer(
        "Show the top 2 houses.",
        "address: 111 Main St, price: 200000\naddress: 222 Main St, price: 900000",
        [{
            "tool": "sql",
            "columns": ["address", "price"],
            "rows": [["111 Main St", "900000"], ["222 Main St", "200000"]],
            "evidence": {"evidence_id": "ev-row-check"},
        }],
    )

    assert result["status"] == "failed"
    assert any(claim["status"] == "failed" for claim in result["claims"])


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
