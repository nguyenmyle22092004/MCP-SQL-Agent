from __future__ import annotations

import json
from pathlib import Path

from app.agents.graph import run_agent


ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "benchmark" / "questions.json"


def main():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    total = 0
    used_sql = 0
    passed = 0
    print("========================================")
    print("HOUSE PRICE SQL AGENT BENCHMARK")
    print("========================================")
    for item in questions:
        result = run_agent(item["question"], "bench_user", f"bench_{item['id']}")
        trace_path = ROOT / "logs" / "traces" / f"{result.get('trace_id')}.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        tool_calls = trace.get("tool_calls", [])
        sql_used = any(call.get("tool") == "execute_sql" for call in tool_calls)
        validation_status = trace.get("validation", {}).get("status", "unknown")
        if sql_used:
            used_sql += 1
        if validation_status == "pass":
            passed += 1
        execution_time = 0
        for event in trace.get("events", []):
            if event.get("event_type") == "SQL_RESULT":
                execution_time = int(event.get("execution_time_ms", 0))
                break
        print(f"{item['id']}  {validation_status.upper()}  {'SQL MCP' if sql_used else 'DIRECT'}  {execution_time}ms")
        total += 1
    print(f"\n{total}/{total} completed")
    print(f"{used_sql}/{total} used SQL MCP")
    print(f"{passed}/{total} passed grounding validation")


if __name__ == "__main__":
    main()
