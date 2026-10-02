from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.exc import OperationalError

from app.agents.graph import run_agent
from app.api.schemas import ChatRequest, ChatResponse

router = APIRouter()
TRACE_VIEW_PATH = Path(__file__).resolve().parents[1] / "web" / "trace.html"


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


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
    )


@router.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict:
    trace_path = Path(__file__).resolve().parents[2] / "logs" / "traces" / f"{trace_id}.json"
    if not trace_path.exists():
        raise HTTPException(status_code=404, detail="Trace not found")
    return json.loads(trace_path.read_text(encoding="utf-8"))


@router.get("/traces/{trace_id}/view", include_in_schema=False)
def view_trace(trace_id: str) -> FileResponse:
    trace_path = Path(__file__).resolve().parents[2] / "logs" / "traces" / f"{trace_id}.json"
    if not trace_path.exists():
        raise HTTPException(status_code=404, detail="Trace not found")
    return FileResponse(TRACE_VIEW_PATH, media_type="text/html")
