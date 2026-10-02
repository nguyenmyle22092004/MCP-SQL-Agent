from sqlalchemy import text

from app.db.database import SessionLocal
from app.mcp.client import MCPClient
from app.mcp.registry import get_mcp_registry


def test_sql_mcp_list_describe_and_execute():
    client = MCPClient(get_mcp_registry()["sql"])
    client.connect()
    try:
        tools = client.list_tools()
        assert any(item.get("name") == "list_tables" for item in tools)
        assert any(item.get("name") == "describe_table" for item in tools)
        assert any(item.get("name") == "execute_sql" for item in tools)

        table_info = client.call_tool("describe_table", table_name="house_prices")
        assert table_info["table"] == "house_prices"
        assert "city" in [col["name"] for col in table_info["columns"]]

        direct_count = SessionLocal().execute(text("SELECT COUNT(*) FROM house_prices")).scalar_one()
        mcp_result = client.call_tool("execute_sql", sql="SELECT COUNT(*) AS total FROM house_prices;")
        assert mcp_result["row_count"] == 1
        assert mcp_result["rows"][0][0] == direct_count
    finally:
        client.close()


def test_sql_security_rejects_dangerous_operations():
    client = MCPClient(get_mcp_registry()["sql"])
    client.connect()
    try:
        for sql in [
            "DELETE FROM house_prices;",
            "UPDATE house_prices SET city = 'X';",
            "DROP TABLE house_prices;",
            "ALTER TABLE house_prices ADD COLUMN temp INT;",
            "INSERT INTO house_prices (address, city, state, bedrooms, bathrooms, area_sqft, year_built, price) VALUES ('a', 'Boston', 'MA', 1, 1.0, 1200, 2020, 300000);",
            "TRUNCATE TABLE house_prices;",
        ]:
            try:
                client.call_tool("execute_sql", sql=sql)
                assert False, f"Expected dangerous SQL to be rejected: {sql}"
            except Exception:
                pass
    finally:
        client.close()
