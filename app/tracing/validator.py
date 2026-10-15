from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import NamedTuple
from uuid import uuid4

from sqlglot import exp, parse


class NumToken(NamedTuple):
    raw: str
    value: Decimal
    start: int
    end: int
    exponent: int
    precision: int


_SCALE_SUFFIXES = r"billion|million|thousand|bn|k|m|b|tỷ|triệu|nghìn"
_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])"
    r"(?P<number>"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|"
    r"\d{1,3}(?:\.\d{3}){2,}|"
    r"\d+(?:[.,]\d+)?"
    r")"
    rf"(?P<suffix>\s*(?:{_SCALE_SUFFIXES}))?"
    r"(?![\w])",
    re.IGNORECASE,
)
_ORDINAL_PREFIX = re.compile(r"^\s*(?:[-*]\s*)?\d+[.)]\s*")
_CONTEXT_ROLE = re.compile(r"^\s*(user|assistant|system|developer)\s*:\s*(.*)$", re.IGNORECASE)
_SCALE_EXPONENTS = {
    "": 0,
    "k": 3,
    "thousand": 3,
    "nghìn": 3,
    "m": 6,
    "million": 6,
    "triệu": 6,
    "b": 9,
    "bn": 9,
    "billion": 9,
    "tỷ": 9,
}


def _numeric_tokens(text: str) -> list[NumToken]:
    tokens = []
    for match in _NUMBER_PATTERN.finditer(text):
        number = match.group("number")
        suffix = (match.group("suffix") or "").strip().lower()
        exponent = _SCALE_EXPONENTS[suffix]
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", number):
            normalized = number.replace(",", "")
        elif re.fullmatch(r"\d{1,3}(?:\.\d{3}){2,}", number):
            normalized = number.replace(".", "")
        elif "," in number:
            normalized = number.replace(",", ".")
        else:
            normalized = number
        precision = len(normalized.partition(".")[2])
        try:
            value = Decimal(normalized) * (Decimal(10) ** exponent)
        except InvalidOperation:
            continue
        tokens.append(NumToken(match.group(), value, match.start(), match.end(), exponent, precision))
    return tokens


def extract_numeric_values(text: str) -> list[float]:
    if not text:
        return []
    return [float(token.value) for token in _numeric_tokens(text)]


def _matches_value(claim: NumToken, value: Decimal) -> bool:
    precision_tolerance = max(
        Decimal("0.01"),
        Decimal("0.5") * (Decimal(10) ** (claim.exponent - claim.precision)),
    )
    tolerance_cap = max(Decimal("0.01"), Decimal("0.05") * abs(value))
    tolerance = min(precision_tolerance, tolerance_cap)
    return abs(claim.value - value) <= tolerance


def _row_numeric_values(row: list | tuple, columns: list[str]) -> list[Decimal]:
    values = []
    for column, cell in zip(columns, row):
        if str(column).lower() == "address" or cell is None:
            continue
        values.extend(token.value for token in _numeric_tokens(str(cell)))
    return values


def _address_mentions(line: str, row_records: list[dict]) -> list[tuple[int, int, dict]]:
    mentions = []
    for record in row_records:
        address = record.get("address")
        if not address:
            continue
        mentions.extend(
            (match.start(), match.end(), record)
            for match in re.finditer(rf"(?<!\w){re.escape(str(address))}(?!\w)", line, re.IGNORECASE)
        )
    return sorted(mentions, key=lambda mention: (mention[0], mention[1]))


def _strip_ordinal(line: str) -> str:
    return _ORDINAL_PREFIX.sub("", line, count=1)


def _tokens_outside_addresses(line: str, mentions: list[tuple[int, int, dict]]) -> list[NumToken]:
    return [
        token for token in _numeric_tokens(line)
        if not any(start <= token.start and token.end <= end for start, end, _ in mentions)
    ]


def _user_context_text(recent_context: str) -> str:
    lines = recent_context.splitlines()
    has_roles = any(_CONTEXT_ROLE.match(line) for line in lines)
    if not has_roles:
        return recent_context

    user_lines = []
    current_role = ""
    for line in lines:
        match = _CONTEXT_ROLE.match(line)
        if match:
            current_role = match.group(1).lower()
            if current_role == "user":
                user_lines.append(match.group(2))
        elif current_role == "user":
            user_lines.append(line)
    return "\n".join(user_lines)


def _sql_has_aggregate(item: dict, aggregate: type[exp.Func]) -> bool | None:
    sql = item.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        return None
    try:
        statements = parse(sql, read="postgres")
    except Exception:
        return None
    if len(statements) != 1 or statements[0] is None:
        return None
    return statements[0].find(aggregate) is not None


def _has_required_aggregate(sql_results: list[dict], aggregate: type[exp.Func], column_tokens: tuple[str, ...]) -> bool:
    for item in sql_results:
        sql_has_aggregate = _sql_has_aggregate(item, aggregate)
        if sql_has_aggregate is not None:
            if sql_has_aggregate:
                return True
            continue
        if any(
            token in str(column).lower()
            for column in item.get("columns", [])
            for token in column_tokens
        ):
            return True
    return False


def _nearest_address_record(token: NumToken, mentions: list[tuple[int, int, dict]]) -> dict | None:
    if not mentions:
        return None
    token_center = (token.start + token.end) / 2
    return min(
        mentions,
        key=lambda mention: abs(token_center - ((mention[0] + mention[1]) / 2)),
    )[2]


def validate_answer(
    question: str,
    answer: str,
    tool_results: list[dict] | None = None,
    recent_context: str = "",
    trusted_context: str = "",
) -> dict:
    tool_results = tool_results or []
    question_lower = question.lower()
    sql_results = [item for item in tool_results if isinstance(item, dict) and item.get("tool", "sql") == "sql"]
    requires_sql = any(term in question_lower for term in ("how many", "how much", "count", "average", "price", "cost", "cheapest", "expensive", "highest", "lowest", "top ", "bedroom", "houses", "built"))
    if requires_sql:
        if not sql_results:
            return {
                "status": "failed",
                "errors": ["A factual database answer requires an SQL tool result."],
                "expected": [],
                "actual": extract_numeric_values(answer),
                "claims": [],
            }
        if any(not item.get("evidence") for item in sql_results):
            return {
                "status": "failed",
                "errors": ["A factual database answer requires trace-linked SQL evidence."],
                "expected": [],
                "actual": extract_numeric_values(answer),
                "claims": [],
            }

    row_records = []
    evidence_values: list[Decimal] = []
    aggregate_values: list[Decimal] = []
    expected: list[float] = []
    rows_seen = 0
    context_text = "\n".join((question, _user_context_text(recent_context), trusted_context))
    globally_allowed = [token.value for token in _numeric_tokens(context_text)]
    for item in sql_results:
        columns = item.get("columns", [])
        rows = item.get("rows", [])
        rows_seen += len(rows)
        has_address_column = any(str(column).lower() == "address" for column in columns)
        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue
            row_dict = dict(zip(columns, row))
            row_values = _row_numeric_values(row, columns)
            evidence_values.extend(row_values)
            if not has_address_column:
                aggregate_values.extend(row_values)
            expected.extend(float(value) for value in row_values)
            row_dict["_numeric_values"] = row_values
            row_records.append(row_dict)
        if not rows:
            value = item.get("value")
            if isinstance(value, (int, float, Decimal)):
                numeric_value = Decimal(str(value))
                evidence_values.append(numeric_value)
                aggregate_values.append(numeric_value)
                expected.append(float(value))
        row_count = item.get("row_count")
        if isinstance(row_count, int):
            globally_allowed.append(Decimal(row_count))
        evidence = item.get("evidence") or {}
        evidence_row_count = evidence.get("row_count")
        if isinstance(evidence_row_count, int):
            globally_allowed.append(Decimal(evidence_row_count))
        sql = str(item.get("sql") or evidence.get("sql") or "")
        limit_match = re.search(r"\bLIMIT\s+(\d+)\b", sql, re.IGNORECASE)
        if limit_match:
            globally_allowed.append(Decimal(limit_match.group(1)))

    line_records = []
    for original_line in (part.strip() for part in answer.splitlines() if part.strip()):
        line = _strip_ordinal(original_line)
        mentions = _address_mentions(line, row_records)
        tokens = _tokens_outside_addresses(line, mentions)
        line_records.append((original_line, tokens, mentions))
    answer_tokens = [token for _, tokens, _ in line_records for token in tokens]
    answer_values = [float(token.value) for token in answer_tokens]
    evidence_ids = [
        item.get("evidence", {}).get("evidence_id")
        for item in sql_results
        if item.get("evidence", {}).get("evidence_id")
    ]
    claims = []
    errors = []
    top_match = re.search(r"\btop\s+(\d+)\b", question_lower)
    if top_match and rows_seen and rows_seen < int(top_match.group(1)):
        errors.append(f"Requested top {top_match.group(1)} rows but SQL returned only {rows_seen}.")
    if ("how many" in question_lower or "total number" in question_lower) and "house" in question_lower and not any(word in question_lower for word in ("price", "bedroom", "city", "built", "cost")):
        if not _has_required_aggregate(sql_results, exp.Count, ("count", "house_count", "total")):
            errors.append("A total-house-count question requires a COUNT result column or COUNT(*) query.")
    if ("average" in question_lower or "average price" in question_lower) and sql_results:
        if not _has_required_aggregate(sql_results, exp.Avg, ("avg", "average")):
            errors.append("An average question requires an AVG query or average result column.")
    if not answer_values and expected:
        errors.append("Answer contained no numeric claim even though SQL data was returned.")

    for original_line, line_tokens, mentions in line_records:
        if not line_tokens:
            continue
        unmatched = []
        for token in line_tokens:
            row_record = _nearest_address_record(token, mentions)
            if row_record is None:
                scoped_values = evidence_values
            else:
                scoped_values = row_record["_numeric_values"] + aggregate_values
            if not any(_matches_value(token, value) for value in scoped_values + globally_allowed):
                unmatched.append(token)
        claim_status = "failed" if unmatched else "pass"
        claims.append({
            "claim_id": f"claim_{uuid4().hex[:12]}",
            "kind": "numeric",
            "text": original_line,
            "values": [float(token.value) for token in line_tokens],
            "evidence_ids": evidence_ids,
            "status": claim_status,
        })
        if unmatched:
            reason = "not grounded in the SQL row for the nearest addressed house or allowed context" if mentions else "not grounded in SQL evidence or allowed context"
            errors.append(
                f"Numeric value(s) {', '.join(token.raw for token in unmatched)} {reason}: {original_line[:160]}"
            )

    status = "pass" if not errors and (not expected or bool(answer_values)) else "failed"
    return {"status": status, "errors": errors, "expected": expected, "actual": answer_values, "claims": claims}