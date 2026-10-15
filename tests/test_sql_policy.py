import pytest

from app.db.sql_policy import validate_read_only_sql


@pytest.mark.parametrize(
    "sql",
    [
        "WITH d AS (DELETE FROM house_prices RETURNING *) SELECT * FROM d",
        "SELECT * INTO x FROM house_prices",
        "SELECT * FROM house_prices FOR UPDATE",
        "WITH tables AS (SELECT 1 FROM house_prices) SELECT * FROM information_schema.tables",
        "SELECT query_to_xml('select * from pg_user', true, true, '') FROM house_prices",
        "SELECT pg_sleep(10) FROM house_prices",
        "SELECT nextval('s') FROM house_prices",
        "SELECT * FROM house_prices; DROP TABLE house_prices",
        "SELECT * FROM pg_catalog.pg_tables",
        "SELECT * FROM public.other_table",
        "",
        "SELECT " + (" " * 20001) + "1 FROM house_prices",
    ],
)
def test_read_only_sql_policy_rejects_invalid_queries(sql):
    with pytest.raises(ValueError):
        validate_read_only_sql(sql)


def test_read_only_sql_policy_rejects_non_string_input():
    with pytest.raises(ValueError, match="SQL must be a string"):
        validate_read_only_sql(None)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM house_prices",
        "SELECT AVG(price) FROM public.house_prices WHERE district = 'A'",
        "WITH t AS (SELECT district, AVG(price) AS p FROM house_prices GROUP BY district) SELECT * FROM t ORDER BY p DESC LIMIT 5",
        "SELECT ROUND(AVG(price), 2) FROM house_prices",
        "SELECT 1 FROM house_prices;",
        "SELECT CAST(price AS NUMERIC), CASE WHEN bedrooms > 1 THEN COALESCE(city, 'x') ELSE NULL END FROM house_prices",
        "SELECT district, COUNT(*), SUM(price), MIN(price), MAX(price), ROUND(AVG(price), 2) FROM house_prices GROUP BY district",
        "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price), STDDEV(price), VARIANCE(price) FROM house_prices",
    ],
)
def test_read_only_sql_policy_accepts_allowed_queries(sql):
    expected = sql.strip()
    if expected.endswith(";"):
        expected = expected[:-1].rstrip()

    assert validate_read_only_sql(sql) == expected