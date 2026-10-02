from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agents.nodes import agent_node, answer_node, load_long_term_memory, load_short_term_memory, save_memory
from app.agents.state import AgentState
from app.tracing.tracer import TraceRecorder


def build_workflow(tracer: TraceRecorder):
    graph = StateGraph(AgentState)
    graph.add_node("load_short_term_memory", lambda state: load_short_term_memory(state, tracer))
    graph.add_node("load_long_term_memory", lambda state: load_long_term_memory(state, tracer))
    graph.add_node("agent", lambda state: agent_node(state, tracer))
    graph.add_node("answer", lambda state: answer_node(state, tracer))
    graph.add_node("save_memory", lambda state: save_memory(state, tracer))
    graph.set_entry_point("load_short_term_memory")
    graph.add_edge("load_short_term_memory", "load_long_term_memory")
    graph.add_edge("load_long_term_memory", "agent")
    graph.add_conditional_edges(
        "agent",
        lambda state: "answer" if state.get("tool_results") or state.get("final_answer") else "answer",
        {"answer": "answer"},
    )
    graph.add_edge("answer", "save_memory")
    graph.add_edge("save_memory", END)
    return graph.compile()


def run_agent(question: str, user_id: str, conversation_id: str) -> dict:
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
        "validation_status": "pass",
    }
    workflow = build_workflow(tracer)
    result = workflow.invoke(state)
    tracer.save()
    result["trace_id"] = tracer.trace_id
    return result
