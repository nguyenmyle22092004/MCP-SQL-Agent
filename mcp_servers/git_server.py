from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("git_server")


def _run_git(args: list[str]) -> dict:
    proc = subprocess.run(["git", *args], capture_output=True, text=True, cwd=".", check=False)
    return {"args": args, "stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}


@mcp.tool()
def git_status() -> dict:
    return _run_git(["status", "--short", "--branch"])


@mcp.tool()
def git_diff() -> dict:
    return _run_git(["diff", "--stat"])


@mcp.tool()
def git_log() -> dict:
    return _run_git(["log", "--oneline", "-n", "10"])


async def main() -> None:
    await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
