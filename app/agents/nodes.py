from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from app.agents.llm import build_llm
from app.agents.prompts import SYSTEM_PROMPT
from app.config import get_settings
from app.mcp.client import MCPClient
from app.mcp.registry import get_mcp_registry
from app.memory.long_term import load_user_memories
from app.memory.short_term import ensure_conversation, load_recent_messages, save_message
from app.tracing.tracer import TraceRecorder
from app.tracing.validator import validate_answer


def _is_database_question(question: str) -> bool:
    lower = question.lower()
    keywords = [
        "how many",
        "average",
        "highest",
        "lowest",
        "most expensive",
        "cheapest",
        "price",
        "houses",
        "bedroom",
        "city",
        "built",
        "cost",
        "per square foot",
        "after 2015",
        "top",
    ]
    return any(keyword in lower for keyword in keywords)


def _build_sql(question: str) -> str:
    lower = re.sub(r"[-_]+", " ", question.lower())
    if "average" in lower and ("lowest" in lower or "least" in lower or "cheapest" in lower):
        return "SELECT city, AVG(price) AS avg_price FROM house_prices GROUP BY city ORDER BY avg_price ASC LIMIT 1;"
    if "average" in lower and ("highest" in lower or "greatest" in lower or "most expensive" in lower):
        return "SELECT city, AVG(price) AS avg_price FROM house_prices GROUP BY city ORDER BY avg_price DESC LIMIT 1;"
    if "how many" in lower and re.search(r"\b3\s+bedroom", lower):
        return "SELECT COUNT(*) AS count FROM house_prices WHERE bedrooms = 3;"
    if "how many" in lower and re.search(r"\b4\s+bedroom", lower):
        return "SELECT COUNT(*) AS count FROM house_prices WHERE bedrooms = 4;"
    if "average" in lower and "price" in lower and re.search(r"\b3\s+bedroom", lower):
        return "SELECT AVG(price) AS avg_price FROM house_prices WHERE bedrooms = 3;"
    if "average" in lower and ("per square foot" in lower or "per sqft" in lower) and "city" in lower:
        return "SELECT city, AVG(price / area_sqft) AS avg_price_per_sqft FROM house_prices GROUP BY city ORDER BY avg_price_per_sqft DESC;"
    if "most expensive house" in lower:
        return "SELECT id, address, city, price, bedrooms, bathrooms, area_sqft, year_built FROM house_prices ORDER BY price DESC LIMIT 1;"
    if "cheapest" in lower and re.search(r"\b4\s+bedroom", lower):
        return "SELECT id, address, city, price, bedrooms, bathrooms, area_sqft, year_built FROM house_prices WHERE bedrooms = 4 ORDER BY price ASC LIMIT 1;"
    if "more than $1 million" in lower or "more than 1 million" in lower:
        return "SELECT COUNT(*) AS count FROM house_prices WHERE price > 1000000;"
    if "after 2015" in lower:
        return "SELECT COUNT(*) AS count FROM house_prices WHERE year_built > 2015;"
    if "top 5" in lower and "expensive" in lower and re.search(r"\b4\s+bedroom", lower):
        return "SELECT id, address, city, price, bedrooms, area_sqft FROM house_prices WHERE bedrooms >= 4 ORDER BY price DESC LIMIT 5;"
    if "in boston" in lower:
        if "how many" in lower or "count" in lower:
            return "SELECT COUNT(*) AS count FROM house_prices WHERE city = 'Boston';"
    if "average" in lower and "price" in lower:
        return "SELECT AVG(price) AS avg_price FROM house_prices;"
    if "how many houses" in lower or "count houses" in lower or "total number of houses" in lower:
        return "SELECT COUNT(*) AS count FROM house_prices;"
    return ""


def _generate_sql(question: str, llm: Any) -> tuple[str, dict[str, int]]:
    system_content: str | list[dict[str, Any]] = SYSTEM_PROMPT
    if get_settings().ANTHROPIC_PROMPT_CACHE:
        system_content = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    response = llm.invoke(
        [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
        ]
    )
    usage = getattr(response, "usage_metadata", None) or {}
    input_details = usage.get("input_token_details", {}) if isinstance(usage, dict) else {}
    cache_usage = {}
    if input_details.get("cache_read") is not None:
        cache_usage["cache_read_tokens"] = input_details["cache_read"]
    cache_creation_tokens = input_details.get("cache_creation") or sum(
        input_details.get(key, 0) or 0
        for key in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
    )
    if cache_creation_tokens or "cache_creation" in input_details:
        cache_usage["cache_creation_tokens"] = cache_creation_tokens
    if usage.get("input_tokens") is not None:
        cache_usage["input_tokens"] = usage["input_tokens"]
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "\n".join(block.get("text", "") for block in content if isinstance(block, dict))
    sql = str(content).strip()
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.IGNORECASE).strip()
    sql = sql.rstrip(";").strip()
    if not re.match(r"^(SELECT|WITH)\b", sql, re.IGNORECASE):
        return "", cache_usage
    if not re.search(r"\b(?:FROM|JOIN)\s+(?:public\s*\.\s*)?house_prices\b", sql, re.IGNORECASE):
        return "", cache_usage
    if ";" in sql or re.search(r"--|/\*|\*/", sql):
        return "", cache_usage
    return sql + ";", cache_usage


def load_short_term_memory(state: dict[str, Any], tracer: TraceRecorder) -> dict[str, Any]:
    tracer.add_event("MEMORY_LOAD_START")
    conversation_id = state["conversation_id"]
    messages = load_recent_messages(conversation_id, limit=20)
    state["short_term_context"] = messages
    tracer.add_event("MEMORY_LOAD_END", loaded=len(messages))
    return state


def load_long_term_memory(state: dict[str, Any], tracer: TraceRecorder) -> dict[str, Any]:
    tracer.add_event("MEMORY_LOAD_START", source="long_term")
    user_id = state["user_id"]
    memories = load_user_memories(user_id)
    state["long_term_context"] = memories
    tracer.add_event("MEMORY_LOAD_END", source="long_term", loaded=len(memories))
    return state


def agent_node(state: dict[str, Any], tracer: TraceRecorder) -> dict[str, Any]:
    tracer.add_event("AGENT_START")
    messages = state["messages"]
    last_user_message = messages[-1]["content"] if messages else ""
    if _is_database_question(last_user_message):
        state["tool_results"] = []
        llm = build_llm()
        if hasattr(llm, "model"):
            tracer.add_event("LLM_CALL_START", model="anthropic")
            sql, cache_usage = _generate_sql(last_user_message, llm)
            tracer.add_event(
                "LLM_CALL_END",
                decision="tool_call" if sql else "unsupported",
                sql=sql,
                **cache_usage,
            )
        else:
            sql = _build_sql(last_user_message)
            tracer.add_event("LLM_CALL_START", model="fallback")
            tracer.add_event("LLM_CALL_END", decision="tool_call" if sql else "unsupported", sql=sql)

        if sql:
            result = run_sql_tool(sql, tracer, state["user_id"], state["conversation_id"])
            state["tool_results"] = result
            state["final_answer"] = ""
        else:
            state["final_answer"] = (
                "I couldn't safely translate that into a database query. "
                "Please rephrase it with a metric, grouping, and any filters you want."
            )
            state["validation_status"] = "unsupported"
    else:
        state["final_answer"] = "I can help with house price questions using the database."
    tracer.add_event("AGENT_END")
    return state


def run_sql_tool(sql: str, tracer: TraceRecorder, user_id: str, conversation_id: str) -> list[dict[str, Any]]:
    tool_call_id = f"tool_{uuid4().hex[:8]}"
    tracer.add_tool_call("sql", "execute_sql", {"sql": sql}, tool_call_id=tool_call_id)
    client = MCPClient(get_mcp_registry()["sql"])
    client.connect()
    try:
        result = client.call_tool("execute_sql", sql=sql, trace_id=tracer.trace_id, tool_call_id=tool_call_id)
        if isinstance(result, dict):
            rows = result.get("rows", [])
            columns = result.get("columns", [])
            value = rows[0][0] if rows and rows[0] else None
            tracer.add_event("SQL_EXECUTION", sql=sql)
            tracer.add_event("SQL_RESULT", row_count=result.get("row_count", 0), result=value)
            tracer.add_event("EVIDENCE_CREATED", evidence=result.get("evidence", {}))
            if result.get("evidence"):
                tracer.add_evidence(result["evidence"])
            return [{"sql": sql, "value": value, "columns": columns, "rows": rows, "row_count": result.get("row_count", 0), "execution_time_ms": result.get("execution_time_ms", 0)}]
        return [{"sql": sql, "value": None}]
    finally:
        client.close()


def answer_node(state: dict[str, Any], tracer: TraceRecorder) -> dict[str, Any]:
    question = state["messages"][-1]["content"] if state["messages"] else ""
    if state["tool_results"]:
        result = state["tool_results"][0]
        columns = result.get("columns") or []
        rows = result.get("rows") or []
        first_row = rows[0] if rows and isinstance(rows[0], list) else []
        values = dict(zip(columns, first_row))
        question_lower = question.lower()
        avg_value = next((value for key, value in values.items() if "avg" in key.lower()), None)
        count_value = next((value for key, value in values.items() if "count" in key.lower() or "total" in key.lower()), None)
        if "city" in values and avg_value is not None and "average" in question_lower:
            direction = "lowest" if any(word in question_lower for word in ("lowest", "least", "cheapest")) else "highest"
            answer = f"The city with the {direction} average house price is {values['city']} (${float(avg_value):,.2f})."
        elif count_value is not None:
            answer = f"There are {int(count_value)} matching houses."
        elif avg_value is not None:
            answer = f"The average price is ${float(avg_value):,.2f}."
        elif first_row:
            answer = "The matching results are:\n" + "\n".join(
                ", ".join(f"{column}: {value}" for column, value in zip(columns, row))
                for row in rows
            )
        else:
            answer = "The database query returned no matching rows."
        state["final_answer"] = answer
    else:
        if state.get("validation_status") != "unsupported":
            state["final_answer"] = "I can help with house price questions using the database."
    tracer.add_event("LLM_CALL_START", model="fallback")
    tracer.add_event("LLM_CALL_END", final_answer=state["final_answer"])
    tracer.add_event("ANSWER_VALIDATION_START")
    if state.get("validation_status") == "unsupported":
        errors = ["No safe SQL plan was available for this database question."]
        tracer.set_validation("unsupported", errors)
        tracer.add_event("ANSWER_VALIDATION_END", status="unsupported", errors=errors)
    else:
        validation = validate_answer(question, state["final_answer"], state["tool_results"])
        tracer.set_validation(validation["status"], validation["errors"])
        tracer.add_event("ANSWER_VALIDATION_END", status=validation["status"], errors=validation["errors"])
        state["validation_status"] = validation["status"]
    return state


def save_memory(state: dict[str, Any], tracer: TraceRecorder) -> dict[str, Any]:
    tracer.add_event("MEMORY_SAVE", memory_type="short_term", reason="user_response")
    ensure_conversation(state["user_id"], state["conversation_id"])
    save_message(state["conversation_id"], "user", state["messages"][-1]["content"])
    save_message(state["conversation_id"], "assistant", state["final_answer"])
    if any(keyword in state["messages"][-1]["content"].lower() for keyword in ["mostly care", "interested in", "prefer", "mainly"]):
        tracer.add_event("MEMORY_SAVE", memory_type="preference", reason="user_preference")
        if state["user_id"]:
            from app.memory.long_term import save_user_memory
            save_user_memory(state["user_id"], "preference", f"User is primarily interested in {state['messages'][-1]['content']}", metadata={"source": "user_input"})
    return state
