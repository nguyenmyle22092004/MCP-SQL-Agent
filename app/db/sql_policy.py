from __future__ import annotations

import re

from sqlglot import exp, parse


BLOCKED_FUNCTIONS = {"set_config", "nextval", "setval", "lo_import", "lo_export"}
MODIFYING_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter, exp.Command)


def validate_read_only_sql(sql: str) -> str:
    if not isinstance(sql, str):
        raise ValueError("SQL must be a string.")
    cleaned = sql.strip()
    if not cleaned:
        raise ValueError("SQL cannot be empty.")
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()
    if ";" in cleaned:
        raise ValueError("Multiple SQL statements are not allowed.")
    if "--" in cleaned or "/*" in cleaned or "*/" in cleaned:
        raise ValueError("Comments are not allowed.")
    if len(cleaned) > 20000:
        raise ValueError("SQL is too long.")
    try:
        statements = parse(cleaned, read="postgres")
    except Exception as exc:
        raise ValueError("SQL could not be parsed as PostgreSQL.") from exc
    if len(statements) != 1 or statements[0] is None:
        raise ValueError("Exactly one SQL statement is required.")
    statement = statements[0]
    if not isinstance(statement, exp.Select):
        raise ValueError("Only read-only SELECT statements are allowed.")
    if any(statement.find(node_type) is not None for node_type in MODIFYING_NODES):
        raise ValueError("Data-modifying and administrative SQL is not allowed.")
    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    source_tables = []
    for table in statement.find_all(exp.Table):
        table_name = table.name.lower()
        schema_name = (table.db or "public").lower()
        if table_name in cte_names:
            continue
        source_tables.append(table_name)
        if table_name != "house_prices" or schema_name != "public":
            raise ValueError("Queries may only read from public.house_prices.")
    if "house_prices" not in source_tables:
        raise ValueError("Queries must read from public.house_prices.")
    for function in statement.find_all(exp.Func):
        function_names = {function.sql_name().lower(), str(getattr(function, "name", "")).lower()}
        blocked_function = next(
            (name for name in function_names if name.startswith("pg_") or name in BLOCKED_FUNCTIONS),
            None,
        )
        if blocked_function:
            raise ValueError(f"Function '{blocked_function}' is not allowed.")
    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        raise ValueError("Only SELECT or WITH queries are allowed.")
    return cleaned
