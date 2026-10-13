from __future__ import annotations

import re
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import uuid4

from app.agents.llm import build_llm
from app.agents.prompts import ANSWER_PROMPT, SYSTEM_PROMPT
from app.config import get_settings
from app.db.sql_policy import validate_read_only_sql
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


def _generate_sql(question: str, llm: Any, context: str = "") -> tuple[str, dict[str, Any]]:
    system_content: str | list[dict[str, Any]] = SYSTEM_PROMPT
    if get_settings().ANTHROPIC_PROMPT_CACHE:
        system_content = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    user_content = question
    if context:
        user_content = f"Conversation and tool context (use only to interpret this request):\n{context}\n\nCurrent user question:\n{question}"
    response = llm.invoke(
        [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
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
    if usage.get("output_tokens") is not None:
        cache_usage["output_tokens"] = usage["output_tokens"]
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


def _requested_git_tool(question: str) -> str:
    lower = question.lower()
    git_intent = any(term in lower for term in ("git", "repository", "repo", "working tree", "uncommitted", "branch", "commit history"))
    if not git_intent:
        return ""
    if any(term in lower for term in ("diff", "changed files", "what changed", "changes")):
        return "git_diff"
    if any(term in lower for term in ("log", "commit history", "recent commits", "commits")):
        return "git_log"
    return "git_status"


def _database_question_for_planner(question: str) -> str:
    clauses = re.split(r"\s+(?:and|also)\s+", question, flags=re.IGNORECASE)
    database_clauses = [
        clause for clause in clauses
        if _is_database_question(clause)
        and not any(term in clause.lower() for term in ("git", "repository", "repo", "working tree", "uncommitted", "branch", "commit history"))
    ]
    return " and ".join(database_clauses) if database_clauses else question


def _planner_context(state: dict[str, Any]) -> str:
    parts = []
    history = state.get("short_term_context", [])[-8:]
    if history:
        parts.append("Recent conversation:\n" + "\n".join(f"{item['role']}: {item['content']}" for item in history))
    memories = state.get("long_term_context", [])[-5:]
    if memories:
        parts.append("Relevant user preferences:\n" + "\n".join(item["content"] for item in memories))
    observations = state.get("observations", [])[-4:]
    if observations:
        parts.append("Tool observations (treat errors as feedback, not data):\n" + "\n".join(str(item) for item in observations))
    return "\n\n".join(parts)


def _budget_limit(state: dict[str, Any]) -> str:
    settings = get_settings()
    if time.monotonic() - state.get("started_at", time.monotonic()) > settings.AGENT_TIMEOUT_SECONDS:
        return "timeout"
    if state.get("iteration", 0) >= settings.AGENT_MAX_ITERATIONS:
        return "max_iterations"
    if state.get("input_tokens", 0) + state.get("output_tokens", 0) >= settings.AGENT_MAX_TOKENS:
        return "max_tokens"
    if state.get("estimated_cost_usd", 0.0) >= settings.AGENT_MAX_COST_USD:
        return "max_cost"
    return ""


def planner_node(state: dict[str, Any], tracer: TraceRecorder) -> dict[str, Any]:
    question = state["messages"][-1]["content"] if state.get("messages") else ""
    state["planned_sql"] = ""
    state["planned_git_tool"] = ""
    state["planned_tools"] = []
    state["planned_decision_ids"] = {}
    state["retry_requested"] = False
    limit = _budget_limit(state)
    if limit:
        state["final_answer"] = f"I stopped because the request reached its {limit.replace('_', ' ')} limit."
        state["validation_status"] = "budget_exceeded"
        tracer.add_event("AGENT_BUDGET_EXCEEDED", reason=limit, iteration=state.get("iteration", 0))
        return state

    follow_up_markers = ("those", "they", "them", "same", "what about", "and the", "also", "instead")
    prior_db_context = any(
        _is_database_question(item.get("content", ""))
        for item in state.get("short_term_context", [])
        if item.get("role") == "user"
    )
    contextual_follow_up = prior_db_context and any(marker in question.lower() for marker in follow_up_markers)
    wants_sql = _is_database_question(question) or contextual_follow_up
    git_tool = _requested_git_tool(question)
    if not wants_sql and not git_tool:
        state["final_answer"] = "I can help with house-price questions or read-only Git status, diff, and log requests."
        state["validation_status"] = "not_applicable"
        decision_id = tracer.record_decision(None, "no_tool", "request does not require an available data tool", planner="intent_router")
        tracer.record_policy_check(decision_id, "no_unnecessary_tool_call", "pass")
        return state

    state["iteration"] = state.get("iteration", 0) + 1
    if wants_sql and not any(result.get("tool") == "sql" and not result.get("error") for result in state.get("tool_results", [])):
        sql_question = _database_question_for_planner(question)
        settings = get_settings()
        remaining_seconds = max(0.1, settings.AGENT_TIMEOUT_SECONDS - (time.monotonic() - state.get("started_at", time.monotonic())))
        llm = build_llm(timeout_seconds=remaining_seconds)
        if hasattr(llm, "model"):
            llm_call_id = tracer.start_llm_call(str(llm.model), "sql_planning", iteration=state["iteration"])
            try:
                sql, usage = _generate_sql(sql_question, llm, _planner_context(state))
            except Exception as exc:
                tracer.end_llm_call(llm_call_id, "error", error_type=type(exc).__name__)
                sql = ""
                state["last_error"] = {"type": type(exc).__name__, "message": str(exc)}
            else:
                tracer.end_llm_call(llm_call_id, "success", **usage)
                input_tokens = int(usage.get("input_tokens", 0))
                output_tokens = int(usage.get("output_tokens", 0))
                state["input_tokens"] = state.get("input_tokens", 0) + input_tokens
                state["output_tokens"] = state.get("output_tokens", 0) + output_tokens
                state["estimated_cost_usd"] = state.get("estimated_cost_usd", 0.0) + (
                    input_tokens * settings.LLM_INPUT_COST_PER_MILLION_USD
                    + output_tokens * settings.LLM_OUTPUT_COST_PER_MILLION_USD
                ) / 1_000_000
                if state["input_tokens"] + state["output_tokens"] >= settings.AGENT_MAX_TOKENS:
                    state["last_budget_reason"] = "max_tokens"
                    sql = ""
                elif state["estimated_cost_usd"] >= settings.AGENT_MAX_COST_USD:
                    state["last_budget_reason"] = "max_cost"
                    sql = ""
            policy_error = ""
            if sql:
                try:
                    sql = validate_read_only_sql(sql)
                except ValueError as exc:
                    policy_error = str(exc)
                    sql = ""
            decision_id = tracer.record_decision(
                llm_call_id,
                "call_sql_tool" if sql else "unsupported",
                "model produced a read-only SQL plan" if sql else policy_error or "model returned no allowed SQL plan",
                iteration=state["iteration"],
                sql=sql,
            )
        else:
            sql = _build_sql(sql_question)
            policy_error = ""
            if sql:
                try:
                    sql = validate_read_only_sql(sql)
                except ValueError as exc:
                    policy_error = str(exc)
                    sql = ""
            decision_id = tracer.record_decision(
                None,
                "call_sql_tool" if sql else "unsupported",
                "deterministic fallback matched a supported query" if sql else policy_error or "deterministic fallback has no matching query",
                planner="deterministic_fallback",
                iteration=state["iteration"],
                sql=sql,
            )
        tracer.record_policy_check(
            decision_id,
            "single_read_only_house_prices_query",
            "pass" if sql else "blocked",
            "SQL plan passed PostgreSQL AST and table allowlist checks" if sql else policy_error or "no allowed SQL plan was generated",
        )
        if sql:
            state["planned_sql"] = sql
            state["planned_tools"].append("sql")
            state["planned_decision_ids"]["sql"] = decision_id
        elif not git_tool:
            if state.get("last_budget_reason"):
                limit = state["last_budget_reason"].replace("_", " ")
                state["final_answer"] = f"I stopped before executing a tool because the request reached its {limit} limit."
                state["validation_status"] = "budget_exceeded"
                tracer.add_event("AGENT_BUDGET_EXCEEDED", reason=state["last_budget_reason"], iteration=state["iteration"])
            elif state.get("last_error"):
                state["final_answer"] = "The SQL planner failed before producing a safe query. No database answer was generated."
                state["validation_status"] = "failed"
            else:
                state["final_answer"] = "I couldn't safely translate that into a database query. Please rephrase it with a metric, grouping, and any filters you want."
                state["validation_status"] = "unsupported"

    if git_tool and not any(result.get("tool") == "git" and result.get("tool_name") == git_tool for result in state.get("tool_results", [])):
        state["planned_git_tool"] = git_tool
        state["planned_tools"].append("git")
        decision_id = tracer.record_decision(None, "call_git_tool", "read-only repository inspection requested", tool=git_tool)
        tracer.record_policy_check(decision_id, "git_read_only_tools", "pass", "selected read-only Git operation")
        state["planned_decision_ids"]["git"] = decision_id
    return state


def _is_retryable_sql_error(error: dict[str, Any]) -> bool:
    message = str(error.get("message", "")).lower()
    return any(term in message for term in (
        "syntax error", "column does not exist", "relation does not exist", "operator does not exist",
        "ambiguous column", "invalid input syntax", "datatype mismatch",
    ))


def execute_tools_node(state: dict[str, Any], tracer: TraceRecorder) -> dict[str, Any]:
    jobs = {}
    settings = get_settings()
    remaining_seconds = max(0.1, settings.AGENT_TIMEOUT_SECONDS - (time.monotonic() - state.get("started_at", time.monotonic())))
    tool_timeout = min(settings.MCP_TOOL_TIMEOUT_SECONDS, remaining_seconds)
    if "sql" in state.get("planned_tools", []):
        jobs["sql"] = lambda: run_sql_tool(
            state["planned_sql"], tracer, state["user_id"], state["conversation_id"], state["planned_decision_ids"].get("sql"), tool_timeout
        )[0]
    if "git" in state.get("planned_tools", []):
        jobs["git"] = lambda: run_git_tool(state["planned_git_tool"], tracer, state["planned_decision_ids"].get("git"), tool_timeout)
    if len(jobs) > 1:
        tracer.add_event("TOOL_EXECUTION_PARALLEL_START", tools=list(jobs), reason="independent SQL and Git requests")
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = {name: executor.submit(job) for name, job in jobs.items()}
            results = {name: future.result() for name, future in futures.items()}
        tracer.add_event("TOOL_EXECUTION_PARALLEL_END", tools=list(results))
    else:
        results = {name: job() for name, job in jobs.items()}

    for name, result in results.items():
        result["tool"] = name
        state.setdefault("tool_results", []).append(result)
        observation = {"tool": name, "status": "error" if result.get("error") else "success"}
        observation.update(result.get("error") or {"result": result.get("rows", result.get("stdout", result.get("value")))})
        state.setdefault("observations", []).append(observation)

    sql_error = next((result["error"] for result in results.values() if result.get("tool") == "sql" and result.get("error")), None)
    settings = get_settings()
    within_time = time.monotonic() - state.get("started_at", time.monotonic()) < settings.AGENT_TIMEOUT_SECONDS
    within_budget = (
        state.get("iteration", 0) < settings.AGENT_MAX_ITERATIONS
        and state.get("input_tokens", 0) + state.get("output_tokens", 0) < settings.AGENT_MAX_TOKENS
        and state.get("estimated_cost_usd", 0.0) < settings.AGENT_MAX_COST_USD
    )
    if sql_error and _is_retryable_sql_error(sql_error) and within_time and within_budget:
        state["retry_requested"] = True
        state["last_error"] = sql_error
        tracer.add_event("AGENT_RETRY_SCHEDULED", iteration=state["iteration"], error_type=sql_error.get("type"))
    elif sql_error:
        state["last_error"] = sql_error
        state["validation_status"] = "failed"
    return state


def agent_node(state: dict[str, Any], tracer: TraceRecorder) -> dict[str, Any]:
    """Compatibility alias for the conditional loop's planner node."""
    return planner_node(state, tracer)


def run_sql_tool(
    sql: str,
    tracer: TraceRecorder,
    user_id: str,
    conversation_id: str,
    decision_id: str | None = None,
    timeout_seconds: float | None = None,
) -> list[dict[str, Any]]:
    tool_call_id = f"tool_{uuid4().hex[:8]}"
    tracer.add_tool_call("sql", "execute_sql", {"sql": sql}, tool_call_id=tool_call_id, decision_id=decision_id)
    client = MCPClient(get_mcp_registry()["sql"])
    try:
        client.connect()
        result = client.call_tool(
            "execute_sql",
            timeout_seconds=timeout_seconds or get_settings().MCP_TOOL_TIMEOUT_SECONDS,
            sql=sql,
            trace_id=tracer.trace_id,
            tool_call_id=tool_call_id,
        )
        if isinstance(result, dict):
            rows = result.get("rows", [])
            columns = result.get("columns", [])
            value = rows[0][0] if rows and rows[0] else None
            tracer.add_event("SQL_EXECUTION", sql=sql)
            tracer.add_event("SQL_RESULT", row_count=result.get("row_count", 0), result=value)
            if result.get("evidence"):
                tracer.add_evidence(result["evidence"])
            tracer.end_tool_call(tool_call_id, "success", row_count=result.get("row_count", 0))
            observation_id = tracer.record_observation(
                tool_call_id,
                {"status": "success", "row_count": result.get("row_count", 0), "columns": columns, "rows": rows},
            )
            return [{
                "tool": "sql",
                "sql": sql,
                "value": value,
                "columns": columns,
                "rows": rows,
                "row_count": result.get("row_count", 0),
                "execution_time_ms": result.get("execution_time_ms", 0),
                "evidence": result.get("evidence"),
                "tool_call_id": tool_call_id,
                "observation_id": observation_id,
            }]
        error = {"type": "UnexpectedToolResult", "message": "SQL tool returned an unsupported response."}
        tracer.end_tool_call(tool_call_id, "error", error=error)
        tracer.record_observation(tool_call_id, {"status": "error", **error})
        return [{"tool": "sql", "sql": sql, "value": None, "error": error, "tool_call_id": tool_call_id}]
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, ValueError) or "not allowed" in str(exc).lower() or "only read-only" in str(exc).lower():
            tracer.record_policy_check(decision_id, "sql_server_ast_read_only_allowlist", "blocked", str(exc))
        tracer.end_tool_call(tool_call_id, "error", error=error)
        tracer.record_observation(tool_call_id, {"status": "error", **error})
        return [{"tool": "sql", "sql": sql, "value": None, "error": error, "tool_call_id": tool_call_id}]
    finally:
        client.close()


def run_git_tool(
    tool_name: str,
    tracer: TraceRecorder,
    decision_id: str | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    if tool_name not in {"git_status", "git_diff", "git_log"}:
        return {"tool": "git", "tool_name": tool_name, "error": {"type": "PolicyError", "message": "Git tool is not allowed."}}
    tool_call_id = f"tool_{uuid4().hex[:8]}"
    tracer.add_tool_call("git", tool_name, {}, tool_call_id=tool_call_id, decision_id=decision_id)
    client = MCPClient(get_mcp_registry()["git"])
    try:
        client.connect()
        result = client.call_tool(tool_name, timeout_seconds=timeout_seconds or get_settings().MCP_TOOL_TIMEOUT_SECONDS)
        if not isinstance(result, dict) or result.get("returncode", 1) not in (0, 1):
            error = {
                "type": "GitToolError",
                "message": str(result.get("stderr", "Git command failed")) if isinstance(result, dict) else "Invalid Git response",
            }
            tracer.end_tool_call(tool_call_id, "error", error=error)
            tracer.record_observation(tool_call_id, {"status": "error", **error}, source="git")
            return {"tool": "git", "tool_name": tool_name, "tool_call_id": tool_call_id, "error": error}
        tracer.end_tool_call(tool_call_id, "success", returncode=result.get("returncode", 0))
        observation_id = tracer.record_observation(
            tool_call_id,
            {"status": "success", "stdout": result.get("stdout", ""), "stderr": result.get("stderr", "")},
            source="git",
        )
        return {
            "tool": "git",
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "observation_id": observation_id,
            **result,
        }
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        tracer.end_tool_call(tool_call_id, "error", error=error)
        tracer.record_observation(tool_call_id, {"status": "error", **error}, source="git")
        return {"tool": "git", "tool_name": tool_name, "tool_call_id": tool_call_id, "error": error}
    finally:
        client.close()


def _synthesize_answer_with_llm(
    question: str,
    context: str,
    observations: list[dict[str, Any]],
    state: dict[str, Any],
    tracer: TraceRecorder,
    repair_feedback: str = "",
) -> str | None:
    settings = get_settings()
    synthesis_budget_reason = ""
    if time.monotonic() - state.get("started_at", time.monotonic()) > settings.AGENT_TIMEOUT_SECONDS:
        synthesis_budget_reason = "timeout"
    elif state.get("input_tokens", 0) + state.get("output_tokens", 0) >= settings.AGENT_MAX_TOKENS:
        synthesis_budget_reason = "max_tokens"
    elif state.get("estimated_cost_usd", 0.0) >= settings.AGENT_MAX_COST_USD:
        synthesis_budget_reason = "max_cost"
    if synthesis_budget_reason:
        tracer.add_event("LLM_SYNTHESIS_SKIPPED", reason=synthesis_budget_reason)
        return None
    remaining_seconds = max(0.1, settings.AGENT_TIMEOUT_SECONDS - (time.monotonic() - state.get("started_at", time.monotonic())))
    llm = build_llm(timeout_seconds=remaining_seconds)
    if not hasattr(llm, "model"):
        return None
    purpose = "answer_repair" if repair_feedback else "answer_synthesis"
    llm_call_id = tracer.start_llm_call(str(llm.model), purpose)
    user_payload = {
        "question": question,
        "conversation_context": context,
        "tool_observations": observations,
    }
    if repair_feedback:
        user_payload["repair_feedback"] = repair_feedback
    try:
        response = llm.invoke([
            {"role": "system", "content": ANSWER_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, default=str)},
        ])
    except Exception as exc:
        tracer.end_llm_call(llm_call_id, "error", error_type=type(exc).__name__)
        tracer.add_event("ANSWER_SYNTHESIS_FALLBACK", reason=type(exc).__name__)
        return None

    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    state["input_tokens"] = state.get("input_tokens", 0) + input_tokens
    state["output_tokens"] = state.get("output_tokens", 0) + output_tokens
    state["estimated_cost_usd"] = state.get("estimated_cost_usd", 0.0) + (
        input_tokens * settings.LLM_INPUT_COST_PER_MILLION_USD
        + output_tokens * settings.LLM_OUTPUT_COST_PER_MILLION_USD
    ) / 1_000_000
    tracer.end_llm_call(llm_call_id, "success", input_tokens=input_tokens, output_tokens=output_tokens)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "\n".join(block.get("text", "") for block in content if isinstance(block, dict))
    answer = str(content).strip()
    return answer or None


def _format_grounded_sql_answer(question: str, result: dict[str, Any]) -> str:
    columns = result.get("columns") or []
    rows = result.get("rows") or []
    formatted_rows = [
        ", ".join(f"{column}: {value}" for column, value in zip(columns, row))
        for row in rows
        if isinstance(row, (list, tuple))
    ]
    if not formatted_rows:
        return "The database query returned no matching rows."

    answer = "Matching results:\n" + "\n".join(formatted_rows)
    question_lower = question.lower()
    wants_recommendation = any(
        phrase in question_lower
        for phrase in ("best", "recommend", "choose", "recommendation")
    )
    if wants_recommendation and {"address", "bedrooms", "bathrooms"}.issubset(columns):
        candidates = [dict(zip(columns, row)) for row in rows if isinstance(row, (list, tuple))]
        cheapest_requested = any(
            phrase in question_lower for phrase in ("cheapest", "lowest price", "lowest-priced")
        )
        if cheapest_requested:
            recommended = candidates[0]
            recommendation_basis = "the lowest listed price"
        else:
            recommended = max(
                candidates,
                key=lambda row: (
                    float(row.get("bedrooms") or 0),
                    float(row.get("bathrooms") or 0),
                ),
            )
            recommendation_basis = "the listed bedroom count, with bathroom count as a tie-breaker"
        property_name = ", ".join(
            str(recommended[key]) for key in ("address", "city") if recommended.get(key) is not None
        )
        if property_name:
            answer += (
                f"\n\nBased on {recommendation_basis}, the recommended option is {property_name}."
                f"\nIts database row lists {recommended['bedrooms']} bedrooms and "
                f"{recommended['bathrooms']} bathrooms."
                "\nThis recommendation uses only those listed room counts; the results do not include "
                "property condition or your personal priorities."
            )
    return answer


def answer_node(state: dict[str, Any], tracer: TraceRecorder) -> dict[str, Any]:
    question = state["messages"][-1]["content"] if state["messages"] else ""
    tool_results = state.get("tool_results", [])
    sql_result = next(
        (result for result in reversed(tool_results) if result.get("tool") == "sql" and not result.get("error")),
        None,
    )
    git_results = [result for result in tool_results if result.get("tool") == "git"]
    if sql_result:
        result = sql_result
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
        else:
            answer = _format_grounded_sql_answer(question, result)
        state["final_answer"] = answer
    else:
        if state.get("validation_status") not in {"unsupported", "budget_exceeded", "failed"}:
            state["final_answer"] = "I can help with house price questions using the database."
            if not any(event.get("event_type") == "AGENT_DECISION" for event in tracer.data["events"]):
                decision_id = tracer.record_decision(None, "no_tool", "request does not require a database lookup", planner="intent_router")
                tracer.record_policy_check(decision_id, "no_unnecessary_tool_call", "pass")
            state["validation_status"] = "not_applicable"
    if git_results:
        git_sections = []
        for result in git_results:
            git_output = result.get("stdout") or result.get("error", {}).get("message", "No output")
            git_sections.append(f"{result.get('tool_name', 'Git')} result:\n{git_output.strip()}")
        prefix = state.get("final_answer", "").strip()
        state["final_answer"] = "\n\n".join(part for part in [prefix, *git_sections] if part)
    if sql_result or git_results:
        observations = []
        if sql_result:
            observations.append({
                "tool": "sql.execute_sql",
                "tool_call_id": sql_result.get("tool_call_id"),
                "evidence_id": (sql_result.get("evidence") or {}).get("evidence_id"),
                "columns": sql_result.get("columns", []),
                "rows": sql_result.get("rows", []),
            })
        for git_result in git_results:
            observations.append({
                "tool": f"git.{git_result.get('tool_name', 'unknown')}",
                "tool_call_id": git_result.get("tool_call_id"),
                "stdout": git_result.get("stdout", ""),
                "error": git_result.get("error"),
            })
        synthesized = _synthesize_answer_with_llm(question, _planner_context(state), observations, state, tracer)
        if synthesized:
            state["final_answer"] = synthesized
            answer_source = "anthropic_observations"
        else:
            answer_source = "deterministic_formatter"
    else:
        answer_source = "deterministic_no_tool"
    tracer.add_event("ANSWER_SYNTHESIS", source=answer_source, has_tool_results=bool(tool_results))
    tracer.add_event("ANSWER_VALIDATION_START")
    if state.get("validation_status") == "unsupported":
        errors = ["No safe SQL plan was available for this database question."]
        tracer.set_validation("unsupported", errors)
        tracer.add_event("ANSWER_VALIDATION_END", status="unsupported", errors=errors)
    elif state.get("validation_status") == "budget_exceeded":
        errors = ["The agent stopped because a request budget was exhausted before producing a grounded answer."]
        tracer.set_validation("budget_exceeded", errors)
        tracer.add_event("ANSWER_VALIDATION_END", status="budget_exceeded", errors=errors)
    elif state.get("validation_status") == "not_applicable":
        tracer.set_validation("not_applicable")
        tracer.add_event("ANSWER_VALIDATION_END", status="not_applicable", errors=[])
    elif _is_database_question(question) and sql_result is None:
        error_result = next((result.get("error") for result in reversed(tool_results) if result.get("tool") == "sql" and result.get("error")), None)
        errors = [error_result.get("message", "Required SQL evidence is missing.")] if error_result else ["Required SQL evidence is missing."]
        state["validation_status"] = "failed"
        tracer.set_validation("failed", errors)
        tracer.add_event("ANSWER_VALIDATION_END", status="failed", errors=errors)
    elif sql_result is None:
        tool_errors = [result.get("error") for result in tool_results if result.get("error")]
        if tool_errors:
            state["final_answer"] = "I couldn't complete the requested tool operation. No verified result is available."
            errors = [f"{error.get('type', 'ToolError')}: {error.get('message', 'tool failed')}" for error in tool_errors]
            state["validation_status"] = "failed"
            tracer.set_validation("failed", errors)
            tracer.add_event("ANSWER_VALIDATION_END", status="failed", errors=errors)
        else:
            state["validation_status"] = "not_applicable"
            tracer.set_validation("not_applicable")
            tracer.add_event("ANSWER_VALIDATION_END", status="not_applicable", errors=[])
    else:
        evidence = sql_result.get("evidence") or {}
        if evidence.get("trace_id") != tracer.trace_id or evidence.get("tool_call_id") != sql_result.get("tool_call_id"):
            error = "SQL evidence is missing or is not linked to this trace and tool call."
            validation = {"status": "failed", "errors": [error], "claims": [{"kind": "provenance", "text": error, "evidence_ids": [], "status": "failed"}]}
        else:
            recent_context = "\n".join(
                f"{item.get('role', 'user')}: {item.get('content', '')}"
                for item in state.get("short_term_context", [])[-8:]
            )
            validation = validate_answer(question, state["final_answer"], [sql_result], recent_context=recent_context)
            tool_errors = [
                f"{result.get('tool_name', result.get('tool', 'tool'))} failed: {result['error'].get('message', 'unknown error')}"
                for result in tool_results
                if result.get("error")
            ]
            if tool_errors:
                validation["status"] = "failed"
                validation["errors"].extend(tool_errors)
            if validation["status"] == "failed" and synthesized and not tool_errors:
                rejected_errors = validation["errors"]
                repair_feedback = (
                    "Rewrite the previous answer once. These numeric claims were rejected: "
                    + "; ".join(rejected_errors)
                    + ". Use only numbers present in the SQL evidence, the current question, or recent conversation. "
                    "For each named house, keep its factual numbers from that house's single row. "
                    "Do not claim that a home is suitable or has enough space for a family."
                )
                tracer.add_event("ANSWER_REPAIR_START", rejected_errors=rejected_errors)
                repaired = _synthesize_answer_with_llm(
                    question,
                    _planner_context(state),
                    observations,
                    state,
                    tracer,
                    repair_feedback=repair_feedback,
                )
                if repaired:
                    repaired_validation = validate_answer(
                        question,
                        repaired,
                        [sql_result],
                        recent_context=recent_context,
                    )
                    if tool_errors:
                        repaired_validation["status"] = "failed"
                        repaired_validation["errors"].extend(tool_errors)
                    tracer.add_event(
                        "ANSWER_REPAIR_END",
                        status=repaired_validation["status"],
                        errors=repaired_validation["errors"],
                    )
                    if repaired_validation["status"] == "pass":
                        state["final_answer"] = repaired
                        validation = repaired_validation
                else:
                    tracer.add_event("ANSWER_REPAIR_END", status="skipped", errors=[])
                if validation["status"] != "pass":
                    state["final_answer"] = synthesized
                grounded_answer = _format_grounded_sql_answer(question, sql_result)
                grounded_validation = validate_answer(
                    question,
                    grounded_answer,
                    [sql_result],
                    recent_context=recent_context,
                )
                if validation["status"] != "pass" and grounded_validation["status"] == "pass":
                    state["final_answer"] = grounded_answer
                    validation = grounded_validation
                    tracer.add_event(
                        "ANSWER_SYNTHESIS_FALLBACK",
                        reason="answer_validation_failed",
                        rejected_errors=rejected_errors,
                    )
        for claim in validation.get("claims", []):
            tracer.add_claim(claim)
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
