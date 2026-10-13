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
    iteration: int
    max_iterations: int
    started_at: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    planned_sql: str
    planned_git_tool: str
    planned_tools: list[str]
    planned_decision_ids: dict[str, str]
    observations: list[dict]
    retry_requested: bool
    last_error: dict
