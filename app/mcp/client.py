from __future__ import annotations

import asyncio
import json
import os

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class MCPClient:
    def __init__(self, config):
        self.config = config
        self._connected = False

    def connect(self):
        self._connected = True
        return self

    def _run_request(self, request):
        if not self._connected:
            raise RuntimeError("MCP client is not connected.")

        async def _request_async():
            params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                env={**os.environ, **self.config.environment},
                cwd=os.getcwd(),
            )
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return await request(session)

        return asyncio.run(_request_async())

    def list_tools(self):
        async def _list_tools(session):
            result = await session.list_tools()
            return [tool.model_dump() for tool in result.tools]
        return self._run_request(_list_tools)

    def call_tool(self, name: str, **kwargs):
        async def _call_tool(session):
            result = await session.call_tool(name, arguments=kwargs)
            if getattr(result, "isError", getattr(result, "is_error", False)):
                details = [getattr(item, "text", "") for item in getattr(result, "content", [])]
                raise RuntimeError("MCP tool call failed: " + " ".join(filter(None, details)))
            structured_content = getattr(result, "structuredContent", None)
            if structured_content is None:
                structured_content = getattr(result, "structured_content", None)
            if structured_content is not None:
                return structured_content
            if hasattr(result, "content"):
                text_content = [getattr(item, "text", None) for item in result.content]
                if len(text_content) == 1 and text_content[0]:
                    try:
                        return json.loads(text_content[0])
                    except json.JSONDecodeError:
                        pass
                content = []
                for item in result.content:
                    if hasattr(item, "model_dump"):
                        content.append(item.model_dump())
                    else:
                        content.append(str(item))
                return content
            return result
        return self._run_request(_call_tool)

    def close(self):
        self._connected = False
