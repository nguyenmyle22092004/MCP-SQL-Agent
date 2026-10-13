from __future__ import annotations

import time

from langgraph.graph import END, StateGraph

from app.agents.nodes import answer_node, execute_tools_node, load_long_term_memory, load_short_term_memory, planner_node, save_memory
from app.agents.state import AgentState
from app.config import get_settings
from app.tracing.tracer import TraceRecorder


def build_workflow(tracer: TraceRecorder):
    graph = StateGraph(AgentState)
    graph.add_node("load_short_term_memory", lambda state: load_short_term_memory(state, tracer))
    graph.add_node("load_long_term_memory", lambda state: load_long_term_memory(state, tracer))
    graph.add_node("planner", lambda state: planner_node(state, tracer))
    graph.add_node("execute_tools", lambda state: execute_tools_node(state, tracer))
    graph.add_node("answer", lambda state: answer_node(state, tracer))
    graph.add_node("save_memory", lambda state: save_memory(state, tracer))
    graph.set_entry_point("load_short_term_memory")
    graph.add_edge("load_short_term_memory", "load_long_term_memory")
    graph.add_edge("load_long_term_memory", "planner")
    graph.add_conditional_edges(
        "planner",
        lambda state: "execute_tools" if state.get("planned_tools") else "answer",
        {"execute_tools": "execute_tools", "answer": "answer"},
    )
    graph.add_conditional_edges(
        "execute_tools",
        lambda state: "planner" if state.get("retry_requested") else "answer",
        {"planner": "planner", "answer": "answer"},
    )
    graph.add_edge("answer", "save_memory")
    graph.add_edge("save_memory", END)
    return graph.compile()


def run_agent(question: str, user_id: str, conversation_id: str) -> dict:
    settings = get_settings()
    tracer = TraceRecorder(user_id=user_id, conversation_id=conversation_id, message=question)
    state: AgentState = {
        "messages": [{"role": "user", "content": question}],
        "user_id": user_id,
        "conversation_id": conversation_id,
        "short_term_context": [],
        "long_term_context": [],
        "tool_results": [],
        "trace_id": tracer.trace_id,
        "final_answer": "",
        "validation_status": "pending",
        "iteration": 0,
        "max_iterations": settings.AGENT_MAX_ITERATIONS,
        "started_at": time.monotonic(),
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
        "planned_sql": "",
        "planned_git_tool": "",
        "planned_tools": [],
        "planned_decision_ids": {},
        "observations": [],
        "retry_requested": False,
        "last_error": {},
    }
    workflow = build_workflow(tracer)
    try:
        result = workflow.invoke(state, config={"recursion_limit": settings.AGENT_MAX_ITERATIONS * 4 + 20})
    except Exception as exc:
        tracer.add_event("AGENT_RUN_FAILED", error_type=type(exc).__name__, message=str(exc))
        tracer.set_validation("failed", ["Agent execution failed before a grounded answer was produced."])
        tracer.save()
        raise
    tracer.add_event(
        "AGENT_RUN_FINISHED",
        iterations=result.get("iteration", 0),
        input_tokens=result.get("input_tokens", 0),
        output_tokens=result.get("output_tokens", 0),
        estimated_cost_usd=result.get("estimated_cost_usd", 0.0),
        elapsed_ms=int((time.monotonic() - result.get("started_at", time.monotonic())) * 1000),
    )
    tracer.save()
    result["trace_id"] = tracer.trace_id
    return result
