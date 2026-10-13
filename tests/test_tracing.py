import json
from pathlib import Path

from app.agents.graph import run_agent
from app.agents.nodes import run_sql_tool
from app.tracing.tracer import TraceRecorder


def test_trace_recorder_correlates_decision_tool_and_observation():
    tracer = TraceRecorder(user_id="trace-user", conversation_id="trace-conv", message="question")
    llm_call_id = tracer.start_llm_call("claude", "planning")
    decision_id = tracer.record_decision(llm_call_id, "call_tool", "database question")
    tracer.record_policy_check(decision_id, "read_only_sql", "pass")
    tool_call_id = tracer.add_tool_call("sql", "execute_sql", {"sql": "SELECT 1"}, decision_id=decision_id)
    tracer.end_tool_call(tool_call_id, "success", row_count=1)
    observation_id = tracer.record_observation(tool_call_id, {"rows": [[1]]})
    evidence_id = tracer.add_evidence({"trace_id": tracer.trace_id, "tool_call_id": tool_call_id})
    tracer.end_llm_call(llm_call_id, "success", decision_id=decision_id)

    events = tracer.data["events"]
    assert tracer.data["llm_calls"][0]["llm_call_id"] == llm_call_id
    assert any(event.get("decision_id") == decision_id for event in events)
    assert any(event.get("policy_check_id") for event in events)
    assert any(event.get("tool_call_id") == tool_call_id and event["event_type"] == "MCP_CALL_END" for event in events)
    observation = next(event for event in events if event["event_type"] == "TOOL_OBSERVATION")
    assert observation["observation_id"] == observation_id
    assert tracer.data["evidence"][0]["evidence_id"] == evidence_id
    assert tracer.data["evidence"][0]["tool_call_id"] == tool_call_id
    assert tracer.data["tool_calls"][0]["decision_id"] == decision_id
    assert tracer.data["tool_calls"][0]["status"] == "success"
    assert not tracer.validate_contract()


def test_trace_contract_rejects_unclosed_call_and_unlinked_evidence():
    tracer = TraceRecorder()
    tool_call_id = tracer.add_tool_call("sql", "execute_sql", {"sql": "SELECT 1"})
    tracer.add_evidence({"trace_id": "another-trace", "tool_call_id": "missing-tool"})

    errors = tracer.validate_contract()

    assert any("no end event" in error for error in errors)
    assert any("different trace" in error for error in errors)
    assert any("missing tool call" in error for error in errors)


def test_trace_contract_requires_events_to_match_actual_tool_calls():
    tracer = TraceRecorder()
    decision_id = tracer.record_decision(None, "call_sql_tool", "database request")
    tracer.record_policy_check(decision_id, "read_only_sql", "pass")
    tool_call_id = tracer.add_tool_call(
        "sql", "execute_sql", {"sql": "SELECT 1 FROM public.house_prices"}, decision_id=decision_id
    )
    tracer.end_tool_call(tool_call_id, "success", row_count=1)
    tracer.record_observation(tool_call_id, {"status": "success", "rows": [[1]]})
    tracer.add_evidence({"trace_id": tracer.trace_id, "tool_call_id": tool_call_id})
    tracer.data["events"] = [
        event for event in tracer.data["events"]
        if event["event_type"] not in {"MCP_CALL_END", "TOOL_OBSERVATION"}
    ]

    errors = tracer.validate_contract()

    assert any("no end event" in error.lower() for error in errors)
    assert any("no observation" in error.lower() for error in errors)


def test_no_tool_trace_satisfies_trace_contract():
    tracer = TraceRecorder(message="Hello")
    decision_id = tracer.record_decision(None, "no_tool", "no data lookup required")
    tracer.record_policy_check(decision_id, "no_unnecessary_tool_call", "pass")

    assert tracer.validate_contract() == []


def test_trace_contract_requires_policy_event_for_every_decision():
    tracer = TraceRecorder()
    tracer.record_decision(None, "no_tool", "no data lookup required")

    errors = tracer.validate_contract()

    assert any("no policy check" in error.lower() for error in errors)


def test_sql_tool_call_records_linked_request_observation_and_evidence(monkeypatch):
    class FakeMCPClient:
        def __init__(self, config):
            self.config = config

        def connect(self):
            pass

        def call_tool(self, tool_name, **kwargs):
            return {
                "columns": ["count"],
                "rows": [[1]],
                "row_count": 1,
                "execution_time_ms": 2,
                "evidence": {
                    "evidence_id": "ev-sql-test",
                    "trace_id": kwargs["trace_id"],
                    "tool_call_id": kwargs["tool_call_id"],
                    "columns": ["count"],
                    "result": [[1]],
                },
            }

        def close(self):
            pass

    monkeypatch.setattr("app.agents.nodes.MCPClient", FakeMCPClient)
    monkeypatch.setattr("app.agents.nodes.get_mcp_registry", lambda: {"sql": {}})
    tracer = TraceRecorder(message="How many houses?")
    decision_id = tracer.record_decision(None, "call_sql_tool", "database question")
    tracer.record_policy_check(decision_id, "read_only_sql", "pass")

    result = run_sql_tool(
        "SELECT COUNT(*) AS count FROM public.house_prices",
        tracer,
        "test-user",
        "test-conversation",
        decision_id,
    )

    assert tracer.validate_contract() == []
    assert result[0]["tool_call_id"] == tracer.data["tool_calls"][0]["tool_call_id"]
    assert result[0]["observation_id"] == next(
        event["observation_id"] for event in tracer.data["events"] if event["event_type"] == "TOOL_OBSERVATION"
    )
    assert [event["evidence_id"] for event in tracer.data["events"] if event["event_type"] == "EVIDENCE_CREATED"] == ["ev-sql-test"]


def test_tracing_and_grounding_validation():
    result = run_agent("How many 3-bedroom houses are there?", "trace_user", "trace_conv")
    trace_file = Path(__file__).resolve().parent.parent / "logs" / "traces" / f"{result['trace_id']}.json"
    assert trace_file.exists()
    trace = json.loads(trace_file.read_text(encoding="utf-8"))
    assert trace["trace_id"] == result["trace_id"]
    assert trace["validation"]["status"] in {"pass", "failed"}

    bad_result = {"final_answer": "There are 99 houses.", "tool_results": [{"value": 42}]}
    from app.tracing.validator import validate_answer

    validation = validate_answer("How many houses are there?", bad_result["final_answer"], bad_result["tool_results"])
    assert validation["status"] == "failed"
