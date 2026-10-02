from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "traces"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class TraceRecorder:
    def __init__(self, user_id: str = "unknown", conversation_id: str = "unknown", message: str = ""):
        self.trace_id = f"trace_{uuid4().hex[:8]}"
        self.data = {
            "trace_id": self.trace_id,
            "request": {"user_id": user_id, "conversation_id": conversation_id, "message": message},
            "events": [],
            "tool_calls": [],
            "evidence": [],
            "claims": [],
            "validation": {"status": "pass", "errors": []},
        }

    def add_event(self, event_type: str, **payload):
        self.data["events"].append({"event_type": event_type, **payload})

    def add_tool_call(
        self,
        server: str,
        tool: str,
        payload: dict | None = None,
        tool_call_id: str | None = None,
    ):
        record = {"server": server, "tool": tool, "tool_call_id": tool_call_id, "payload": payload or {}}
        self.data["tool_calls"].append(record)
        self.add_event("MCP_CALL_START", server=server, tool=tool, payload=payload or {})
        return tool_call_id

    def add_evidence(self, evidence: dict):
        self.data["evidence"].append(evidence)

    def add_claim(self, claim: str):
        self.data["claims"].append(claim)

    def set_validation(self, status: str, errors: list[str] | None = None):
        self.data["validation"] = {"status": status, "errors": errors or []}

    def save(self):
        path = LOG_DIR / f"{self.trace_id}.json"
        path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        return self.trace_id
