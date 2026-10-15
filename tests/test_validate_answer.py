import pytest

from app.tracing.validator import validate_answer


def _sql_result(columns, rows, *, row_count=None, sql=None, evidence=True):
    result = {
        "tool": "sql",
        "columns": columns,
        "rows": rows,
        "row_count": row_count if row_count is not None else len(rows),
    }
    if sql is not None:
        result["sql"] = sql
    if evidence:
        result["evidence"] = {"evidence_id": "ev-characterization", "row_count": result["row_count"]}
    return result


@pytest.mark.parametrize(
    ("question", "answer", "tool_results", "recent_context", "expected_status"),
    [
        (
            "Show this house price.",
            "7398 Oak St costs $1,612,085.25 and has 6 bedrooms.",
            [_sql_result(["address", "price", "bedrooms"], [["7398 Oak St", "1612085.25", 6]])],
            "",
            "pass",
        ),
        (
            "Show the houses.",
            "7398 Oak St costs $1,607,972.45.",
            [_sql_result(
                ["address", "price"],
                [["7398 Oak St", "1612085.25"], ["5049 River St", "1607972.45"]],
            )],
            "",
            "failed",
        ),
        (
            "Show this house price.",
            "7398 Oak St costs $620,000.",
            [_sql_result(["address", "price"], [["7398 Oak St", "1612085.25"]])],
            "",
            "failed",
        ),
        (
            "How many houses are there?",
            "There are 42 houses.",
            [],
            "",
            "failed",
        ),
        (
            "What is the house price?",
            "The price is $100.",
            [_sql_result(["price"], [["100"]], evidence=False)],
            "",
            "failed",
        ),
        (
            "Show the top 5 houses.",
            "Top 5 houses are returned.",
            [_sql_result(["id", "price"], [[1, 100], [2, 200], [3, 300]])],
            "",
            "failed",
        ),
        (
            "How many houses are there?",
            "There are 3 houses.",
            [_sql_result(["count"], [[3]])],
            "",
            "pass",
        ),
        (
            "What is the average house price?",
            "The average price is $250,000.74.",
            [_sql_result(["avg_price"], [["250000.738333"]])],
            "",
            "pass",
        ),
        (
            "Show this house price.",
            "7398 Oak St costs about $1.2M.",
            [_sql_result(["address", "price"], [["7398 Oak St", "1200000"]])],
            "",
            "pass",
        ),
        (
            "Show this house price.",
            "7398 Oak St costs $1,612,085.25.",
            [_sql_result(["address", "price"], [["7398 Oak St", "1612085.25"]])],
            "",
            "pass",
        ),
    ],
)
def test_validate_answer_characterization(question, answer, tool_results, recent_context, expected_status):
    result = validate_answer(question, answer, tool_results, recent_context)

    assert result["status"] == expected_status
    assert set(result) == {"status", "errors", "expected", "actual", "claims"}


def test_two_address_line_grounds_each_price_to_its_nearest_house():
    result = validate_answer(
        "Show the house prices.",
        "2542 Lake St: $295,620.06 is the first listed result. The next listing is 7419 Pine St: $303,661.18",
        [_sql_result(
            ["address", "price"],
            [["2542 Lake St", "295620.06"], ["7419 Pine St", "303661.18"]],
        )],
    )

    assert result["status"] == "pass"


def test_swapped_prices_for_two_addresses_fail():
    result = validate_answer(
        "Show the house prices.",
        "2542 Lake St: $303,661.18 is the first listed result. The next listing is 7419 Pine St: $295,620.06",
        [_sql_result(
            ["address", "price"],
            [["2542 Lake St", "295620.06"], ["7419 Pine St", "303661.18"]],
        )],
    )

    assert result["status"] == "failed"
    assert any("303,661.18" in error or "295,620.06" in error for error in result["errors"])


@pytest.mark.parametrize(
    ("answer", "price"),
    [
        ("7398 Oak St costs $620,000.", "1612085.25"),
        ("7398 Oak St costs $1M.", "600000"),
    ],
)
def test_fabricated_price_or_overly_broad_suffix_tolerance_fails(answer, price):
    result = validate_answer(
        "Show this house price.",
        answer,
        [_sql_result(["address", "price"], [["7398 Oak St", price]])],
    )

    assert result["status"] == "failed"


@pytest.mark.parametrize(
    ("answer", "price"),
    [
        ("7398 Oak St costs $3.5 million.", "3500000"),
        ("7398 Oak St costs $1.2B.", "1200000000"),
        ("7398 Oak St costs about $1.2M.", "1234567"),
    ],
)
def test_scale_words_and_suffixes_match_evidence(answer, price):
    result = validate_answer(
        "Show this house price.",
        answer,
        [_sql_result(["address", "price"], [["7398 Oak St", price]])],
    )

    assert result["status"] == "pass"


def test_number_from_unvalidated_assistant_context_is_not_allowed():
    result = validate_answer(
        "Show this house.",
        "7398 Oak St has 620,000 bedrooms.",
        [_sql_result(["address", "bedrooms"], [["7398 Oak St", 6]])],
        recent_context="user: Tell me about the property.\nassistant: It has 620,000 features.",
    )

    assert result["status"] == "failed"
    assert any("620,000" in error for error in result["errors"])


def test_explicitly_trusted_validated_assistant_context_is_allowed():
    result = validate_answer(
        "Show this house.",
        "7398 Oak St was discussed for 5 people.",
        [_sql_result(["address", "bedrooms"], [["7398 Oak St", 6]])],
        recent_context="user: Tell me about the property.",
        trusted_context="Previously validated assistant answer: family size 5.",
    )

    assert result["status"] == "pass"


def test_address_line_can_include_value_from_separate_aggregate_result():
    house_result = _sql_result(["address", "price"], [["7398 Oak St", "1612085.25"]])
    average_result = _sql_result(["avg"], [["1500000"]], sql="SELECT AVG(price) FROM house_prices")

    result = validate_answer(
        "Show the house price and average house price.",
        "7398 Oak St costs $1,612,085.25; the average price is $1,500,000.",
        [house_result, average_result],
    )

    assert result["status"] == "pass"


def test_average_question_requires_avg_in_parseable_sql():
    result = validate_answer(
        "What is the average house price?",
        "The average price is $100.",
        [_sql_result(["avg_price"], [[100]], sql="SELECT price FROM public.house_prices")],
    )

    assert result["status"] == "failed"
    assert any("AVG" in error for error in result["errors"])


def test_unaliased_round_avg_sql_passes_average_check():
    result = validate_answer(
        "What is the average house price?",
        "The average price is $250,000.74.",
        [_sql_result(["round"], [["250000.738333"]], sql="SELECT ROUND(AVG(price), 2) FROM public.house_prices")],
    )

    assert result["status"] == "pass"


def test_count_alias_total_houses_passes_count_check():
    result = validate_answer(
        "How many houses are there?",
        "There are 42 houses.",
        [_sql_result(["total_houses"], [[42]], sql="SELECT COUNT(*) AS total_houses FROM public.house_prices")],
    )

    assert result["status"] == "pass"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("3.5 million", 3500000.0),
        ("1.2B", 1200000000.0),
        ("1.250.000", 1250000.0),
        ("1,5 tỷ", 1500000000.0),
        ("2 triệu", 2000000.0),
        ("5 nghìn", 5000.0),
    ],
)
def test_number_parser_consumes_supported_scale_suffixes(text, expected):
    from app.tracing.validator import extract_numeric_values

    assert extract_numeric_values(text) == [expected]
