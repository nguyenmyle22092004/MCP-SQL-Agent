from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.exc import OperationalError

from app.agents.graph import run_agent
from app.api.schemas import ChatRequest, ChatResponse

router = APIRouter()
TRACE_VIEW_PATH = Path(__file__).resolve().parents[1] / "web" / "trace.html"
TRACE_DASHBOARD_PATH = Path(__file__).resolve().parents[1] / "web" / "traces.html"
TRACE_DIR = Path(__file__).resolve().parents[2] / "logs" / "traces"


def _trace_path(trace_id: str) -> Path:
    if not re.fullmatch(r"trace_[a-fA-F0-9]{8,32}", trace_id):
        raise HTTPException(status_code=400, detail="Invalid trace id")
    return TRACE_DIR / f"{trace_id}.json"


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/traces", include_in_schema=False)
def trace_dashboard() -> FileResponse:
    return FileResponse(TRACE_DASHBOARD_PATH, media_type="text/html")


@router.get("/api/traces")
def list_traces(limit: int = 100) -> dict:
    limit = max(1, min(limit, 500))
    items = []
    for trace_path in sorted(TRACE_DIR.glob("trace_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)[:limit]:
        try:
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        events = trace.get("events", [])
        finish = next((event for event in reversed(events) if event.get("event_type") == "AGENT_RUN_FINISHED"), {})
        failures = [
            event for event in events
            if event.get("event_type") == "AGENT_RUN_FAILED"
            or (event.get("event_type") == "TOOL_OBSERVATION" and event.get("observation", {}).get("status") == "error")
        ]
        items.append({
            "trace_id": trace.get("trace_id", trace_path.stem),
            "created_at": datetime.fromtimestamp(trace_path.stat().st_mtime, timezone.utc).isoformat(),
            "question": trace.get("request", {}).get("message", ""),
            "model": next((call.get("model") for call in trace.get("llm_calls", [])), None),
            "validation_status": trace.get("validation", {}).get("status", "unknown"),
            "tool_calls": len(trace.get("tool_calls", [])),
            "latency_ms": finish.get("elapsed_ms"),
            "estimated_cost_usd": finish.get("estimated_cost_usd"),
            "errors": len(failures) + len(trace.get("validation", {}).get("errors", [])),
        })
    return {"traces": items, "count": len(items)}


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        result = run_agent(payload.message, payload.user_id, payload.conversation_id)
    except OperationalError as exc:
        raise HTTPException(
            status_code=503,
            detail="Database connection failed. Check DATABASE_URL credentials and make sure PostgreSQL is running.",
        ) from exc
    return ChatResponse(
        trace_id=result.get("trace_id", ""),
        conversation_id=payload.conversation_id,
        answer=result.get("final_answer", ""),
        validation_status=result.get("validation_status", "pass"),
        decision_action=result.get("decision_action", "skipped"),
    )


@router.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict:
    trace_path = _trace_path(trace_id)
    if not trace_path.exists():
        raise HTTPException(status_code=404, detail="Trace not found")
    return json.loads(trace_path.read_text(encoding="utf-8"))


@router.get("/traces/{trace_id}/view", include_in_schema=False)
def view_trace(trace_id: str) -> FileResponse:
    trace_path = _trace_path(trace_id)
    if not trace_path.exists():
        raise HTTPException(status_code=404, detail="Trace not found")
    return FileResponse(TRACE_VIEW_PATH, media_type="text/html")
