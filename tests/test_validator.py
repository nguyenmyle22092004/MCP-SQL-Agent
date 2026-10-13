from app.tracing.validator import validate_answer


ATLANTA_COLUMNS = ["address", "city", "price", "bedrooms", "bathrooms", "area_sqft", "year_built"]
ATLANTA_ROWS = [
    ["2542 Lake St", "Atlanta", "295620.06", 6, "2.7", 579, 1964],
    ["7419 Pine St", "Atlanta", "303661.18", 3, "4.7", 555, 1983],
    ["6205 Cedar St", "Atlanta", "314946.40", 1, "4.3", 1263, 2005],
    ["4401 Maple St", "Atlanta", "320000.00", 4, "2.2", 1700, 1979],
    ["2549 Lake St", "Atlanta", "351728.66", 5, "1.0", 1149, 1952],
]
ATLANTA_RESULT = {
    "tool": "sql",
    "columns": ATLANTA_COLUMNS,
    "rows": ATLANTA_ROWS,
    "row_count": 5,
    "sql": "SELECT address, city, price, bedrooms, bathrooms, area_sqft, year_built FROM public.house_prices ORDER BY price ASC LIMIT 5",
    "evidence": {"evidence_id": "ev-atlanta-five", "row_count": 5},
}
ATLANTA_QUESTION = (
    "I want to know top 5 of the lowest houses in Atlanta and choose the best option for me? "
    "My family has 5 peoples"
)


def test_valid_ranked_house_answer_accepts_currency_ordinals_and_question_numbers():
    answer = """Based on the tool observations, here are the top 5 lowest-priced houses in Atlanta:
1. **2542 Lake St** - $295,620.06 | 6 bedrooms, 2.7 bathrooms | 579 sqft | Built 1964
2) **7419 Pine St** - $303,661.18 | 3 bedrooms, 4.7 bathrooms | 555 sqft | Built 1983
3. **6205 Cedar St** - $314,946.40 | 1 bedroom, 4.3 bathrooms | 1,263 sqft | Built 2005
**Recommended option for your family of 5:**
**2542 Lake St, Atlanta** is the best choice based on bedroom count. It has 6 bedrooms and 2.7 bathrooms."""

    result = validate_answer(ATLANTA_QUESTION, answer, [ATLANTA_RESULT])

    assert result["status"] == "pass"
    assert result["errors"] == []
    assert all(claim["status"] == "pass" for claim in result["claims"])


def test_unknown_price_fails_with_the_specific_number():
    answer = "2542 Lake St - $299,999.99, 6 bedrooms and 2.7 bathrooms"

    result = validate_answer(ATLANTA_QUESTION, answer, [ATLANTA_RESULT])

    assert result["status"] == "failed"
    assert any("299,999.99" in error for error in result["errors"])


def test_house_line_cannot_borrow_bedroom_count_from_another_row():
    answer = "2542 Lake St - $295,620.06, 1 bedroom, 2.7 bathrooms"

    result = validate_answer(ATLANTA_QUESTION, answer, [ATLANTA_RESULT])

    assert result["status"] == "failed"
    assert any("1" in error and "addressed house" in error for error in result["errors"])


def test_singular_bedroom_matches_bedrooms_column():
    answer = "1. 6205 Cedar St - $314,946.40 | 1 bedroom, 4.3 bathrooms | 1,263 sqft | Built 2005"

    result = validate_answer(ATLANTA_QUESTION, answer, [ATLANTA_RESULT])

    assert result["status"] == "pass"


def test_number_not_in_question_history_evidence_row_count_or_limit_fails():
    result_data = {
        "tool": "sql",
        "columns": ["address", "city", "bedrooms", "bathrooms"],
        "rows": [["6205 Cedar St", "Atlanta", 1, "4.3"]],
        "row_count": 4,
        "sql": "SELECT address, city, bedrooms, bathrooms FROM public.house_prices",
        "evidence": {"evidence_id": "ev-no-five", "row_count": 4},
    }

    result = validate_answer("Show this Atlanta home.", "It has 5 bathrooms.", [result_data])

    assert result["status"] == "failed"
    assert any("5" in error for error in result["errors"])


def test_limit_and_row_count_values_are_allowed_without_question_numbers():
    result_data = {
        "tool": "sql",
        "columns": ["address", "city", "bedrooms", "bathrooms"],
        "rows": [["6205 Cedar St", "Atlanta", 1, "4.3"]],
        "row_count": 5,
        "sql": "SELECT address, city, bedrooms, bathrooms FROM public.house_prices LIMIT 5",
        "evidence": {"evidence_id": "ev-limit-five", "row_count": 5},
    }

    result = validate_answer("Show houses in Atlanta.", "The query returned 5 houses.", [result_data])

    assert result["status"] == "pass"


def test_numeric_conclusion_passes_only_when_grounded():
    grounded = validate_answer(
        ATLANTA_QUESTION,
        "Recommendation: 2542 Lake St has 6 bedrooms and 2.7 bathrooms.",
        [ATLANTA_RESULT],
    )
    invented = validate_answer(
        ATLANTA_QUESTION,
        "Recommendation: 2542 Lake St has 7 bedrooms and 2.7 bathrooms.",
        [ATLANTA_RESULT],
    )

    assert grounded["status"] == "pass"
    assert invented["status"] == "failed"
    assert any("7" in error for error in invented["errors"])


def test_recent_conversation_numbers_are_allowed_but_not_house_row_values():
    result = validate_answer(
        "Which one is best?",
        "For your family of 5, 2542 Lake St has 6 bedrooms.",
        [ATLANTA_RESULT],
        recent_context="User said: My family has 5 people.",
    )

    assert result["status"] == "pass"


def test_k_suffix_rounding_matches_database_price():
    result = validate_answer(
        ATLANTA_QUESTION,
        "2542 Lake St is about $295.6K.",
        [ATLANTA_RESULT],
    )

    assert result["status"] == "pass"