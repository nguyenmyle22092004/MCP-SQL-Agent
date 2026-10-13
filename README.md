PROMPT_VERSION=sql-planner-v2
# House Price SQL Agent

A production-style Python application that combines FastAPI, LangGraph, LLM orchestration, MCP tools, PostgreSQL, and traceable evidence. The system is designed so the LLM reasons and decides whether to use tools, while MCP performs the actual database and Git access.

## Architecture

FastAPI
 ↓
LangGraph
 ↓
LLM
 ↓
MCP Client
 ↓
SQL MCP
 ↓
PostgreSQL

Git MCP
 ↓
Git

## Memory

### Short-term memory
Short-term memory stores the current conversation in the `conversations` and `messages` tables. This allows the agent to resolve references like "they" or "those houses" based on prior turns in the same conversation.

### Long-term memory
Long-term memory stores persistent user preferences and summary-level knowledge in the `long_term_memories` table. It is used as context, but database results always take precedence.

## MCP

The key design rule is:

LLM ≠ MCP

The LLM can generate SQL, but it never directly knows actual database results. The LLM decides to call a tool; the MCP layer executes the SQL against PostgreSQL and returns the actual result. The LLM then explains the returned data.

Example:

LLM generates:
SELECT COUNT(*) FROM house_prices WHERE bedrooms = 3;

But the LLM does not know the resulting count. MCP executes the query. PostgreSQL returns 42. The LLM then explains 42.

## Tracing

Each request gets a trace id. Every LangGraph node, LLM call, MCP call, memory action, and validation result is recorded in `logs/traces/<trace_id>.json`.

## Running

1. Start PostgreSQL:
```bash
docker compose up -d
```

2. Initialize the schema and seed data:
```bash
python scripts/init_db.py
```

3. Run the CLI agent:
```bash
python scripts/run_agent.py "How many 3-bedroom houses are there?"
```

4. Start the API:
```bash
uvicorn app.main:app --reload
```

5. Run tests:
```bash
pytest -q
```

## LLM configuration

Set environment variables in a `.env` file:

```bash
DATABASE_URL=postgresql+psycopg://agent:agent@localhost:5433/house_price_agent
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-5
ANTHROPIC_API_KEY=your-anthropic-api-key
ANTHROPIC_PROMPT_CACHE=true
READONLY_DATABASE_URL=
AGENT_MAX_ITERATIONS=3
AGENT_TIMEOUT_SECONDS=45
AGENT_MAX_TOKENS=12000
AGENT_MAX_COST_USD=0.05
SQL_TOOL_TIMEOUT_SECONDS=10
```

Create an API key in the Anthropic Console and set it as `ANTHROPIC_API_KEY`. Claude Code subscription/login credentials are not automatically usable as Anthropic API keys. If no API key is configured, the app falls back to a deterministic local stub so development can continue without external credentials.

Anthropic prompt caching is enabled by default for the static SQL-planning system prompt. The user question remains outside the cached block. The first request creates an ephemeral cache entry; later requests can reuse it during Anthropic's cache TTL. Set `ANTHROPIC_PROMPT_CACHE=false` to disable it. Trace `LLM_CALL_END` events include `cache_creation_tokens` and `cache_read_tokens` when the provider returns those usage fields; zero cache-read tokens means that request did not hit an existing cache entry.

The SQL MCP connection uses `READONLY_DATABASE_URL` when set; otherwise it uses `DATABASE_URL` but still marks every query transaction read-only and applies `SQL_TOOL_TIMEOUT_SECONDS`. In production (`APP_ENV=production`), configure `READONLY_DATABASE_URL` or both `READONLY_DB_USER` and `READONLY_DB_PASSWORD`; the SQL MCP server refuses to start rather than silently using the application credentials. Create a dedicated PostgreSQL login role with only `CONNECT` on the application database, `USAGE` on `public`, and `SELECT` on `public.house_prices`, then use that role for the read-only URL. Keep the normal `DATABASE_URL` for schema initialization and writes to conversation memory.

The LangGraph workflow uses a bounded planner/tool loop. `AGENT_MAX_ITERATIONS`, `AGENT_TIMEOUT_SECONDS`, `AGENT_MAX_TOKENS`, and `AGENT_MAX_COST_USD` limit retries and model work. SQL and independent Git status/diff/log requests can execute in parallel; each observation and evidence record is linked to its originating decision and MCP tool call. Retry is limited to SQL syntax/schema/type errors; authentication, policy, and other failures stop without claiming an answer.

Open `/traces` for the recent trace dashboard. Each trace shows decisions, policy checks, model calls, tool requests/observations, evidence, and validation. Run `python benchmark/run_benchmark.py` to write a versioned JSON report under `benchmark/results/` with tool-selection accuracy, SQL assertion accuracy, retry success rate, validation pass rate, and latency. Benchmark reports include `PROMPT_VERSION` and model name for comparisons.

## Data set

The seed script creates a deterministic dataset with 360 synthetic houses across the requested cities and value ranges.

## Security notes

- SQL access is read-only.
- DDL and DML statements are rejected by the SQL MCP server.
- Git MCP is read-only; mutation operations are unavailable.
- Secrets are never stored in trace evidence.

## Observability

Use a trace id to inspect what the user asked, which nodes ran, what SQL was executed, what PostgreSQL returned, what memory was loaded, and whether the answer was validated.
