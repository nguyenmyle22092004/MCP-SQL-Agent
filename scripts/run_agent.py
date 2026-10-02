from __future__ import annotations

import json
from pathlib import Path

from app.agents.graph import run_agent


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the house price SQL agent")
    parser.add_argument("question", nargs="?", default="How many 3-bedroom houses are there?", help="Question for the agent")
    parser.add_argument("--user-id", default="user_001")
    parser.add_argument("--conversation-id", default="conv_001")
    args = parser.parse_args()

    result = run_agent(args.question, args.user_id, args.conversation_id)
    trace_path = Path(__file__).resolve().parent.parent / "logs" / "traces" / f"{result.get('trace_id')}.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    tool_calls = trace.get("tool_calls", [])
    sql = ""
    for call in tool_calls:
        if call.get("tool") == "execute_sql":
            sql = call.get("payload", {}).get("sql", "")
            break

    print("Question:")
    print(args.question)
    print("\nAnswer:")
    print(result.get("final_answer", ""))
    print("\nTrace ID:")
    print(result.get("trace_id"))
    print("\nMCP calls:")
    print(len(tool_calls))
    print("\nSQL:")
    print(sql)
    print("\nValidation:")
    print(result.get("validation_status", "unknown").upper())


if __name__ == "__main__":
    main()
