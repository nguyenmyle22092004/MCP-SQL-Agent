import json
from pathlib import Path

from app.agents.graph import run_agent


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
