from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from app.config import get_settings


class FallbackLLM:
    def invoke(self, messages):
        last = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        if "how many" in last.lower() or "average" in last.lower() or "highest" in last.lower():
            return "I should use the SQL MCP tool to answer this from the database."
        return "I will use the SQL MCP tool and answer from actual database results."


def build_llm(timeout_seconds: float | None = None):
    settings = get_settings()
    if settings.LLM_PROVIDER.lower() != "anthropic":
        raise ValueError("LLM_PROVIDER must be 'anthropic'.")
    if settings.ANTHROPIC_API_KEY:
        return ChatAnthropic(
            model=settings.LLM_MODEL,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            temperature=0,
            timeout=timeout_seconds or settings.AGENT_TIMEOUT_SECONDS,
            max_retries=0,
        )
    return FallbackLLM()
