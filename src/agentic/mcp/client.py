"""MCP client — connects to an MCP server and exposes its tools as BaseTool instances."""

from __future__ import annotations

import json
import uuid
from typing import Any, TYPE_CHECKING

import httpx

from agentic.core.message import ToolResult
from agentic.exceptions import MCPError
from agentic.tools.base import BaseTool

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


class MCPRemoteTool(BaseTool):
    """A tool that delegates execution to a remote MCP server."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        server_url: str,
        timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self._server_url = server_url
        self._timeout = timeout

    async def execute(
        self, arguments: dict[str, Any], context: "AgentContext | None" = None
    ) -> ToolResult:
        tool_call_id = str(uuid.uuid4())
        request = {
            "jsonrpc": "2.0",
            "id": tool_call_id,
            "method": "tools/call",
            "params": {"name": self.name, "arguments": arguments},
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    self._server_url,
                    json=request,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()

            if "error" in data:
                return ToolResult(
                    tool_call_id=tool_call_id,
                    name=self.name,
                    content="",
                    error=data["error"].get("message", "Unknown error"),
                )

            result = data.get("result", {})
            content_blocks = result.get("content", [])
            text = "\n".join(
                b.get("text", "") for b in content_blocks if b.get("type") == "text"
            )
            is_error = result.get("isError", False)
            return ToolResult(
                tool_call_id=tool_call_id,
                name=self.name,
                content=text,
                error=text if is_error else None,
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=tool_call_id,
                name=self.name,
                content="",
                error=str(exc),
            )


class MCPClient:
    """MCP client that discovers tools from an MCP server over HTTP.

    Usage::

        client = MCPClient(server_url="http://localhost:8080")
        tools = await client.list_tools()
        # tools are MCPRemoteTool instances usable in any agent
    """

    def __init__(self, server_url: str, timeout: float = 10.0) -> None:
        self.server_url = server_url
        self.timeout = timeout

    async def _send(self, method: str, params: dict[str, Any] | None = None) -> Any:
        request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params or {},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self.server_url,
                json=request,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        if "error" in data:
            raise MCPError(data["error"].get("message", "MCP error"))
        return data.get("result")

    async def initialize(self) -> dict[str, Any]:
        return await self._send("initialize")

    async def list_tools(self) -> list[MCPRemoteTool]:
        result = await self._send("tools/list")
        tools = []
        for t in result.get("tools", []):
            tools.append(
                MCPRemoteTool(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=t.get("inputSchema", {}),
                    server_url=self.server_url,
                    timeout=self.timeout,
                )
            )
        return tools
