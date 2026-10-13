from __future__ import annotations

import asyncio
import re
import sys
import time
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.mcpserver import MCPServer
from mcp.server.stdio import stdio_server
from sqlalchemy import text

from sqlalchemy import create_engine

from app.config import get_readonly_database_url, get_settings
from app.db.sql_policy import validate_read_only_sql
from app.tracing.evidence import add_evidence

mcp = MCPServer("sql_server")
settings = get_settings()
read_engine = create_engine(get_readonly_database_url(settings), pool_pre_ping=True)

def _validate_sql(sql: str) -> str:
    return validate_read_only_sql(sql)


@mcp.tool()
def list_tables() -> dict:
    with read_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name")
        ).fetchall()
    return {"tables": [row[0] for row in rows]}


@mcp.tool()
def describe_table(table_name: str) -> dict:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name or ""):
        raise ValueError("Invalid table name.")
    with read_engine.connect() as conn:
        exists = conn.execute(
            text("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = :table_name)"),
            {"table_name": table_name},
        ).scalar()
    if not exists:
        raise ValueError(f"Table '{table_name}' does not exist.")
    with read_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' AND table_name = :table_name ORDER BY ordinal_position"
            ),
            {"table_name": table_name},
        ).fetchall()
    return {"table": table_name, "columns": [{"name": row[0], "type": row[1]} for row in rows]}


@mcp.tool()
def execute_sql(sql: str, trace_id: str | None = None, tool_call_id: str | None = None) -> dict:
    sql = _validate_sql(sql)
    start = time.perf_counter()
    trace_id = trace_id or f"trace_{uuid4().hex[:8]}"
    tool_call_id = tool_call_id or f"tool_{uuid4().hex[:8]}"
    with read_engine.connect() as conn:
        with conn.begin():
            conn.execute(text("SET TRANSACTION READ ONLY"))
            conn.execute(text(f"SET LOCAL statement_timeout = '{int(settings.SQL_TOOL_TIMEOUT_SECONDS * 1000)}ms'"))
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = result.fetchall()
    trimmed_rows = [list(row) for row in rows[:100]]
    execution_time_ms = int((time.perf_counter() - start) * 1000)
    payload = {
        "columns": columns,
        "rows": trimmed_rows,
        "row_count": len(rows),
        "execution_time_ms": execution_time_ms,
    }
    evidence = add_evidence(
        trace_id=trace_id,
        tool_call_id=tool_call_id,
        sql=sql,
        execution_time_ms=execution_time_ms,
        row_count=len(rows),
        columns=columns,
        result=trimmed_rows,
    )
    payload["evidence"] = evidence
    return payload


async def main() -> None:
    await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
