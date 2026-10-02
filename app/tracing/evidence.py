from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "traces"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def add_evidence(trace_id: str, tool_call_id: str, sql: str | None, execution_time_ms: int | None, row_count: int | None, columns: list[str] | None, result: object) -> dict:
    evidence_id = f"ev_{uuid4().hex[:8]}"
    record = {
        "evidence_id": evidence_id,
        "trace_id": trace_id,
        "tool_call_id": tool_call_id,
        "sql": sql,
        "execution_time_ms": execution_time_ms,
        "row_count": row_count,
        "columns": columns or [],
        "result": result,
    }
    trace_file = LOG_DIR / f"{trace_id}.json"
    if trace_file.exists():
        data = json.loads(trace_file.read_text(encoding="utf-8"))
    else:
        data = {"events": [], "tool_calls": [], "evidence": [], "claims": [], "validation": {"status": "pass", "errors": []}}
    data.setdefault("evidence", []).append(record)
    trace_file.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return record
