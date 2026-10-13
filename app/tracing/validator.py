from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from uuid import uuid4


_NUMBER_PATTERN = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?[kKmM]?(?![\w])")
_ORDINAL_PREFIX = re.compile(r"^\s*(?:[-*]\s*)?\d+[.)]\s*")


def _parse_number(raw: str) -> Decimal | None:
    suffix = raw[-1:].lower()
    multiplier = {"k": Decimal(1000), "m": Decimal(1000000)}.get(suffix, Decimal(1))
    normalized = raw[:-1] if suffix in {"k", "m"} else raw
    try:
        return Decimal(normalized.replace(",", "")) * multiplier
    except InvalidOperation:
        return None


def _numeric_tokens(text: str) -> list[tuple[str, Decimal, int, int]]:
    return [
        (match.group(), value, match.start(), match.end())
        for match in _NUMBER_PATTERN.finditer(text)
        if (value := _parse_number(match.group())) is not None
    ]


def extract_numeric_values(text: str) -> list[float]:
    if not text:
        return []
    return [float(value) for _, value, _, _ in _numeric_tokens(text)]


def _matches_value(claim: Decimal, raw: str, value: Decimal) -> bool:
    decimal_part = raw.split(".", 1)[1] if "." in raw else ""
    suffix = raw[-1:].lower()
    exponent = {"k": 3, "m": 6}.get(suffix, 0)
    if suffix in {"k", "m"} and decimal_part:
        decimal_part = decimal_part[:-1]
    precision = len(decimal_part)
    tolerance = max(Decimal("0.01"), Decimal("0.5") * (Decimal(10) ** (exponent - precision)))
    return abs(claim - value) <= tolerance


def _row_numeric_values(row: list | tuple, columns: list[str]) -> list[Decimal]:
    values = []
    for column, cell in zip(columns, row):
        if str(column).lower() == "address" or cell is None:
            continue
        values.extend(value for _, value, _, _ in _numeric_tokens(str(cell)))
    return values


def _address_row_matches(line: str, row_records: list[dict]) -> list[dict]:
    matches = []
    for record in row_records:
        address = record.get("address")
        if address and re.search(rf"(?<!\w){re.escape(str(address))}(?!\w)", line, re.IGNORECASE):
            matches.append(record)
    return matches


def _strip_ordinal(line: str) -> str:
    return _ORDINAL_PREFIX.sub("", line, count=1)


def _tokens_outside_addresses(line: str, address_matches: list[dict]) -> list[tuple[str, Decimal, int, int]]:
    address_spans = []
    for record in address_matches:
        address = str(record.get("address", ""))
        address_spans.extend(
            (match.start(), match.end())
            for match in re.finditer(re.escape(address), line, re.IGNORECASE)
        )
    return [
        token for token in _numeric_tokens(line)
        if not any(start <= token[2] and token[3] <= end for start, end in address_spans)
    ]


def validate_answer(
    question: str,
    answer: str,
    tool_results: list[dict] | None = None,
    recent_context: str = "",
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
            }
        if any(not item.get("evidence") for item in sql_results):
            return {
                "status": "failed",
                "errors": ["A factual database answer requires trace-linked SQL evidence."],
                "expected": [],
                "actual": extract_numeric_values(answer),
            }

    row_records = []
    evidence_values: list[Decimal] = []
    expected: list[float] = []
    rows_seen = 0
    globally_allowed: list[Decimal] = [value for _, value, _, _ in _numeric_tokens(question + "\n" + recent_context)]
    for item in sql_results:
        columns = item.get("columns", [])
        rows = item.get("rows", [])
        rows_seen += len(rows)
        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue
            row_dict = dict(zip(columns, row))
            row_values = _row_numeric_values(row, columns)
            evidence_values.extend(row_values)
            expected.extend(float(value) for value in row_values)
            row_dict["_numeric_values"] = row_values
            row_records.append(row_dict)
        if not rows:
            value = item.get("value")
            if isinstance(value, (int, float, Decimal)):
                evidence_values.append(Decimal(str(value)))
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

    answer_tokens = []
    for original_line in (part.strip() for part in answer.splitlines() if part.strip()):
        line = _strip_ordinal(original_line)
        answer_tokens.extend(_tokens_outside_addresses(line, _address_row_matches(line, row_records)))
    answer_values = [float(value) for _, value, _, _ in answer_tokens]
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
        if not any(re.search(r"\b(count|house_count|total)\b", str(column), re.IGNORECASE) for item in sql_results for column in item.get("columns", [])):
            errors.append("A total-house-count question requires a COUNT result column.")
    if ("average" in question_lower or "average price" in question_lower) and sql_results:
        if not any(any(token in str(column).lower() for token in ("avg", "average")) for item in sql_results for column in item.get("columns", [])):
            errors.append("An average question requires an aggregate average result column.")
    if not answer_values and expected:
        errors.append("Answer contained no numeric claim even though SQL data was returned.")

    for original_line in (part.strip() for part in answer.splitlines() if part.strip()):
        line = _strip_ordinal(original_line)
        address_matches = _address_row_matches(line, row_records)
        line_tokens = _tokens_outside_addresses(line, address_matches)
        if not line_tokens:
            continue
        if address_matches:
            scoped_values = address_matches[0]["_numeric_values"] if len(address_matches) == 1 else []
        else:
            scoped_values = evidence_values
        unmatched = [
            raw for raw, claim, _, _ in line_tokens
            if not any(_matches_value(claim, raw, value) for value in scoped_values + globally_allowed)
        ]
        if unmatched and address_matches:
            reason = "not grounded in the SQL row for the addressed house or allowed context"
        elif unmatched:
            reason = "not grounded in SQL evidence or allowed context"
        else:
            reason = ""
        claim_status = "failed" if unmatched else "pass"
        claims.append({
            "claim_id": f"claim_{uuid4().hex[:12]}",
            "kind": "numeric",
            "text": original_line,
            "values": [float(value) for _, value, _, _ in line_tokens],
            "evidence_ids": evidence_ids,
            "status": claim_status,
        })
        if unmatched:
            errors.append(f"Numeric value(s) {', '.join(unmatched)} {reason}: {original_line[:160]}")

    status = "pass" if not errors and (not expected or bool(answer_values)) else "failed"
    return {"status": status, "errors": errors, "expected": expected, "actual": answer_values, "claims": claims}