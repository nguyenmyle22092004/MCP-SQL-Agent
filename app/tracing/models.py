from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceEvidence:
    evidence_id: str
    trace_id: str
    tool_call_id: str
    sql: str | None = None
    execution_time_ms: int | None = None
    row_count: int | None = None
    columns: list[str] | None = None
    result: Any = None
