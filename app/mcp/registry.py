from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ServerConfig:
    name: str
    transport: str
    command: str
    args: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)


def get_mcp_registry() -> dict[str, ServerConfig]:
    project_root = Path(__file__).resolve().parents[2]
    python_path = os.pathsep.join(filter(None, [str(project_root), os.environ.get("PYTHONPATH")]))
    return {
        "sql": ServerConfig(
            name="sql",
            transport="stdio",
            command=sys.executable,
            args=["mcp_servers/sql_server.py"],
            environment={"PYTHONPATH": python_path},
        ),
        "git": ServerConfig(
            name="git",
            transport="stdio",
            command=sys.executable,
            args=["mcp_servers/git_server.py"],
            environment={"PYTHONPATH": python_path},
        ),
    }
