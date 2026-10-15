from __future__ import annotations

import re
from typing import Any

from app.tracing.validator import extract_numeric_values, validate_answer

# Twenty scalar features. Bit-packed flags retain boolean signals without a wide vector.
FEATURE_NAMES = (
    "n_numeric_claims",
    "n_unmatched_claims",
    "frac_unmatched",
    "max_relative_error",
    "frac_claims_matched_globally",
    "n_addresses_in_answer",
    "n_cities_not_in_evidence",
    "n_states_not_in_evidence",
    "n_addresses_not_in_evidence",
    "city_state_pair_inconsistent",
    "row_count",
    "n_columns",
    "sql_shape_flags",
    "answer_length_chars",
    "answer_unit_currency_flags",
    "question_location_flags",
    "n_vague_terms",
    "question_has_number",
    "question_type_code",
    "missing_metric",
)
FEATURE_DESCRIPTIONS = {
    "n_numeric_claims": "Number of numeric answer claims returned by validate_answer.",
    "n_unmatched_claims": "Number of claims the deterministic validator marks failed.",
    "frac_unmatched": "Failed numeric claims divided by all claims.",
    "max_relative_error": "Maximum over claims of minimum relative distance to any evidence number.",
    "frac_claims_matched_globally": "Fraction matching question/context values but no SQL-row number.",
    "n_addresses_in_answer": "Count of gazetteer street addresses; nonzero also means has_address_line.",
    "n_cities_not_in_evidence": "Answer cities absent from SQL result city values.",
    "n_states_not_in_evidence": "Answer states absent from SQL result state values.",
    "n_addresses_not_in_evidence": "Answer addresses absent from SQL result address values.",
    "city_state_pair_inconsistent": "A mentioned city is paired with a state other than its gazetteer state.",
    "row_count": "Total returned SQL row count.",
    "n_columns": "Distinct result columns returned by SQL tools.",
    "sql_shape_flags": "Bit mask: LIMIT=1, GROUP BY=2, AVG/COUNT=4.",
    "answer_length_chars": "Answer character length.",
    "answer_unit_currency_flags": "Bit mask: scale unit present=1, currency symbol present=2.",
    "question_location_flags": "Bit mask: city mentioned=1, state mentioned=2.",
    "n_vague_terms": "Count of vague request terms, with expensive counted only without a metric.",
    "question_has_number": "Whether the user question contains a parsed numeric token.",
    "question_type_code": "Keyword category: count=0, average=1, ranking=2, lookup=3, comparison=4, other=5.",
    "missing_metric": "How much/many request has no price/bedroom/sqft/area/year/build metric.",
}

_VAGUE_TERMS = ("cheap", "big", "old", "new", "good", "nice", "large", "small", "recent", "expensive")
_METRIC_WORDS = ("price", "cost", "bedroom", "bathroom", "sqft", "square foot", "area", "year", "built")
_QUESTION_TYPES = ("count", "average", "ranking", "lookup", "comparison", "other")


def _contains_entity(text: str, entity: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(entity)}(?!\w)", text, re.IGNORECASE))


def _as_set(gazetteer: dict[str, Any], key: str) -> set[str]:
    value = gazetteer.get(key, [])
    return {str(item) for item in value}


def _row_numbers(tool_results: list[dict]) -> list[float]:
    values = []
    for result in tool_results:
        columns = result.get("columns", [])
        for row in result.get("rows", []):
            if not isinstance(row, (list, tuple)):
                continue
            values.extend(
                number
                for column, cell in zip(columns, row)
                if str(column).lower() != "address"
                for number in extract_numeric_values(str(cell))
            )
    return values


def _relative_error(claim: float, evidence: list[float]) -> float:
    if not evidence:
        return 1.0
    return min(abs(claim - value) / max(abs(value), 1.0) for value in evidence)


def _question_type(question: str) -> int:
    lower = question.lower()
    if any(term in lower for term in ("how many", "count", "number of")):
        return 0
    if "average" in lower or "mean" in lower:
        return 1
    if any(term in lower for term in ("top", "cheapest", "most expensive", "highest", "lowest", "best")):
        return 2
    if any(term in lower for term in ("which house", "show", "find", "list", "address")):
        return 3
    if any(term in lower for term in ("compare", "versus", " vs ", "difference")):
        return 4
    return 5


def extract_features(
    question: str,
    answer: str,
    tool_results: list[dict],
    recent_context: str,
    trusted_context: str,
    gazetteer: dict[str, Any],
) -> dict[str, float]:
    """Extract the documented 20-dimensional feature vector; has no DB/filesystem access.

    Numeric features summarize validator claims, row evidence, and question/context grounding.
    Entity features compare answer mentions with evidence rows and the supplied gazetteer.
    Evidence and answer-shape features describe the tool response and output form. The
    ``sql_shape_flags`` bits are LIMIT=1, GROUP BY=2, AVG/COUNT=4; the
    ``answer_unit_currency_flags`` bits are unit=1 and currency symbol=2; and
    ``question_location_flags`` bits are city=1 and state=2. A nonzero address count
    represents the requested has-address-line boolean.
    Question features encode entities, vague wording, numeric constraints, a compact type code,
    and whether a measurement/metric is missing.
    """
    validation = validate_answer(question, answer, tool_results, recent_context, trusted_context)
    claims = validation.get("claims", [])
    numeric_claims = [float(value) for claim in claims for value in claim.get("values", [])]
    unmatched_claims = [claim for claim in claims if claim.get("status") == "failed"]
    evidence_numbers = _row_numbers(tool_results)
    global_numbers = extract_numeric_values("\n".join((question, recent_context, trusted_context)))
    relative_errors = [_relative_error(value, evidence_numbers) for value in numeric_claims]
    global_only = [
        value for value in numeric_claims
        if any(abs(value - allowed) < 0.01 for allowed in global_numbers)
        and not any(abs(value - item) < 0.01 for item in evidence_numbers)
    ]

    cities = _as_set(gazetteer, "cities")
    states = _as_set(gazetteer, "states")
    streets = _as_set(gazetteer, "street_names")
    city_state = {str(k): str(v) for k, v in gazetteer.get("city_state", {}).items()}
    lower_answer = answer.lower()
    lower_question = question.lower()
    evidence_rows = [
        dict(zip(item.get("columns", []), row))
        for item in tool_results
        for row in item.get("rows", [])
        if isinstance(row, (list, tuple))
    ]
    evidence_cities = {str(row["city"]) for row in evidence_rows if row.get("city") is not None}
    evidence_states = {str(row["state"]) for row in evidence_rows if row.get("state") is not None}
    evidence_addresses = {str(row["address"]).lower() for row in evidence_rows if row.get("address")}

    answer_cities = {city for city in cities if _contains_entity(answer, city)}
    answer_states = {state for state in states if _contains_entity(answer, state)}
    address_pattern = re.compile(
        r"(?<!\w)\d+\s+(?:[A-Za-z]+\s+)*(?:" + "|".join(re.escape(s) for s in sorted(streets, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    ) if streets else re.compile(r"(?!)")
    answer_addresses = {match.group().strip().lower() for match in address_pattern.finditer(answer)}
    n_cities_unmatched = sum(city not in evidence_cities for city in answer_cities)
    n_states_unmatched = sum(state not in evidence_states for state in answer_states)
    n_addresses_unmatched = sum(address not in evidence_addresses for address in answer_addresses)
    city_mentions = [
        (match.start(), match.end(), city)
        for city in answer_cities
        for match in re.finditer(rf"(?<!\w){re.escape(city)}(?!\w)", answer, re.IGNORECASE)
    ]
    state_mentions = [
        (match.start(), match.end(), state)
        for state in answer_states
        for match in re.finditer(rf"(?<!\w){re.escape(state)}(?!\w)", answer, re.IGNORECASE)
    ]
    inconsistent_pair = False
    for city_start, city_end, city in city_mentions:
        if state_mentions and city in city_state:
            nearest_state = min(
                state_mentions,
                key=lambda item: abs((item[0] + item[1]) / 2 - (city_start + city_end) / 2),
            )[2]
            if nearest_state.upper() != city_state[city].upper():
                inconsistent_pair = True

    columns = {str(column).lower() for item in tool_results for column in item.get("columns", [])}
    sql_text = "\n".join(str(item.get("sql") or (item.get("evidence") or {}).get("sql") or "") for item in tool_results).lower()
    lower_q = question.lower()
    has_metric = any(term in lower_q for term in _METRIC_WORDS)
    vague_count = sum(
        len(re.findall(rf"\b{re.escape(word)}\b", lower_q))
        for word in _VAGUE_TERMS
        if word != "expensive" or not has_metric
    )
    asks_quantity = any(term in lower_q for term in ("how much", "how many"))
    missing_metric = asks_quantity and not any(term in lower_q for term in _METRIC_WORDS)
    question_city = any(_contains_entity(question, city) for city in cities)
    question_state = any(_contains_entity(question, state) for state in states)
    query_type = _question_type(question)
    unit_present = bool(re.search(r"(?:\d\s*(?:k|m|b|bn)\b|\b(?:million|billion|thousand))", lower_answer))
    currency_present = "$" in answer
    sql_has_limit = bool(re.search(r"\blimit\s+\d+", sql_text))
    sql_has_group_by = "group by" in sql_text
    sql_has_avg_or_count = bool(re.search(r"\b(?:avg|count)\s*\(", sql_text))

    features = {
        "n_numeric_claims": float(len(numeric_claims)),
        "n_unmatched_claims": float(len(unmatched_claims)),
        "frac_unmatched": len(unmatched_claims) / max(len(claims), 1),
        "max_relative_error": max(relative_errors, default=0.0),
        "frac_claims_matched_globally": len(global_only) / max(len(numeric_claims), 1),
        "n_addresses_in_answer": float(len(answer_addresses)),
        "n_cities_not_in_evidence": float(n_cities_unmatched),
        "n_states_not_in_evidence": float(n_states_unmatched),
        "n_addresses_not_in_evidence": float(n_addresses_unmatched),
        "city_state_pair_inconsistent": float(inconsistent_pair),
        "row_count": float(sum(int(item.get("row_count", len(item.get("rows", [])))) for item in tool_results)),
        "n_columns": float(len(columns)),
        # Bit flags: LIMIT=1, GROUP BY=2, AVG/COUNT=4.
        "sql_shape_flags": float(sql_has_limit + 2 * sql_has_group_by + 4 * sql_has_avg_or_count),
        "answer_length_chars": float(len(answer)),
        # Bit flags: scale word/suffix=1, currency symbol=2.
        "answer_unit_currency_flags": float(unit_present + 2 * currency_present),
        # Bit flags: city=1, state=2.
        "question_location_flags": float(question_city + 2 * question_state),
        "n_vague_terms": float(vague_count),
        "question_has_number": float(bool(extract_numeric_values(question))),
        "question_type_code": float(query_type),
        "missing_metric": float(missing_metric),
    }
    return {name: float(features[name]) for name in FEATURE_NAMES}
