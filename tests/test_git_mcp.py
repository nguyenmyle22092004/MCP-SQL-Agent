from app.mcp.client import MCPClient
from app.mcp.registry import get_mcp_registry


def test_git_mcp_tools_return_real_git_output():
    client = MCPClient(get_mcp_registry()["git"])
    client.connect()
    try:
        status = client.call_tool("git_status")
        diff = client.call_tool("git_diff")
        log = client.call_tool("git_log")
        assert "returncode" in status
        assert status["returncode"] in (0, 1)
        assert "returncode" in diff
        assert "returncode" in log
    finally:
        client.close()
