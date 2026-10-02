from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    messages: list[dict]
    user_id: str
    conversation_id: str
    short_term_context: list[dict]
    long_term_context: list[dict]
    tool_results: list[dict]
    trace_id: str
    final_answer: str
    validation_status: str
