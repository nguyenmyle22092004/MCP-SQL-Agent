from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from app.agents.graph import run_agent
from app.config import get_settings


ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "benchmark" / "questions.json"


def main():
    settings = get_settings()
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    cases = []
    print("========================================")
    print("HOUSE PRICE SQL AGENT BENCHMARK")
    print("========================================")
    for item in questions:
        started = time.perf_counter()
        run_error = None
        result = {}
        trace = {}
        try:
            result = run_agent(item["question"], "bench_user", f"bench_{item['id']}")
            trace_path = ROOT / "logs" / "traces" / f"{result.get('trace_id')}.json"
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
        except Exception as exc:
            run_error = {"type": type(exc).__name__, "message": str(exc)}
        tool_calls = trace.get("tool_calls", [])
        selected_tools = [call.get("tool") for call in tool_calls]
        sql_calls = [call for call in tool_calls if call.get("tool") == "execute_sql"]
        sql_used = bool(sql_calls)
        expected_tools = item.get("expected_tools", [item.get("expected_tool")] if item.get("expected_tool") else [])
        tool_selection_correct = all(expected_tool in selected_tools for expected_tool in expected_tools)
        sql = str(sql_calls[-1].get("payload", {}).get("sql", "")) if sql_calls else ""
        sql_upper = re.sub(r"\s+", " ", sql).upper()
        expected_fragments = [fragment.upper() for fragment in item.get("expected_sql_contains", [])]
        sql_correct = bool(sql) and all(re.sub(r"\s+", " ", fragment) in sql_upper for fragment in expected_fragments)
        validation_status = trace.get("validation", {}).get("status", "failed" if run_error else "unknown")
        retry_count = sum(event.get("event_type") == "AGENT_RETRY_SCHEDULED" for event in trace.get("events", []))
        sql_result_events = [event for event in trace.get("events", []) if event.get("event_type") == "SQL_RESULT"]
        retries_succeeded = bool(retry_count and sql_result_events and validation_status == "pass")
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        case = {
            "id": item["id"],
            "question": item["question"],
            "trace_id": result.get("trace_id"),
            "expected_tool": item.get("expected_tool"),
            "selected_tools": selected_tools,
            "tool_selection_correct": tool_selection_correct,
            "sql": sql,
            "sql_correct": sql_correct if expected_fragments else None,
            "retry_count": retry_count,
            "retry_succeeded": retries_succeeded if retry_count else None,
            "validation_status": validation_status,
            "latency_ms": elapsed_ms,
            "error": run_error,
        }
        cases.append(case)
        print(f"{item['id']}  {validation_status.upper()}  tool={'OK' if tool_selection_correct else 'MISS'}  sql={'OK' if sql_correct else 'MISS'}  {elapsed_ms}ms")

    total = len(cases)
    retried_cases = [case for case in cases if case["retry_count"]]
    metrics = {
        "cases": total,
        "tool_selection_accuracy": sum(case["tool_selection_correct"] for case in cases) / total if total else 0.0,
        "sql_accuracy": sum(case["sql_correct"] is True for case in cases) / sum(case["sql_correct"] is not None for case in cases) if any(case["sql_correct"] is not None for case in cases) else 0.0,
        "retry_success_rate": sum(case["retry_succeeded"] is True for case in retried_cases) / len(retried_cases) if retried_cases else None,
        "validation_pass_rate": sum(case["validation_status"] == "pass" for case in cases) / total if total else 0.0,
        "mean_latency_ms": sum(case["latency_ms"] for case in cases) / total if total else 0.0,
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.LLM_MODEL if settings.ANTHROPIC_API_KEY else "deterministic_fallback",
        "prompt_version": settings.PROMPT_VERSION,
        "metrics": metrics,
        "results": cases,
    }
    result_dir = ROOT / "benchmark" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{settings.PROMPT_VERSION}.json"
    result_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nMetrics")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print(f"Report: {result_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
