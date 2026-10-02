from __future__ import annotations

import re


def extract_numeric_values(text: str) -> list[float]:
    if not text:
        return []
    matches = re.findall(r"\d[\d,]*\.?\d*", text.replace("$", "").replace(",", ""))
    return [float(v) for v in matches]


def validate_answer(question: str, answer: str, tool_results: list[dict] | None = None) -> dict:
    tool_results = tool_results or []
    expected = []
    for item in tool_results:
        if isinstance(item, dict):
            for row in item.get("rows", []):
                if isinstance(row, (list, tuple)):
                    for value in row:
                        if isinstance(value, (int, float)):
                            expected.append(float(value))
                        elif isinstance(value, str):
                            expected.extend(extract_numeric_values(value))
            if not item.get("rows"):
                value = item.get("value")
                if isinstance(value, (int, float)):
                    expected.append(float(value))
    answer_values = extract_numeric_values(answer)
    errors = []
    if not answer_values and expected:
        errors.append("Answer contained no numeric claim even though SQL data was returned.")
    if expected and answer_values:
        for answer_value in answer_values:
            if not any(round(answer_value, 2) == round(expected_value, 2) for expected_value in expected):
                errors.append(f"Answer value {answer_value} was not present in the SQL result.")
                break
    status = "pass" if not errors else "failed"
    return {"status": status, "errors": errors, "expected": expected, "actual": answer_values}
