"""Model Context Protocol (MCP) server base implementation.

MCP standardizes how LLM applications expose tools and resources to models.
This implements a lightweight MCP server that wraps BaseTool instances.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from agentic.tools.base import BaseTool


class MCPToolDescriptor(BaseModel):
    """MCP-compatible tool descriptor."""

    name: str
    description: str
    inputSchema: dict[str, Any]


class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] = {}


class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: Any | None = None
    error: dict[str, Any] | None = None


class MCPServer:
    """Lightweight MCP server that exposes agentic tools via JSON-RPC 2.0.

    Supports methods:
    - ``tools/list``: enumerate available tools
    - ``tools/call``: execute a tool by name
    """

    def __init__(self, name: str = "agentic-mcp-server", version: str = "0.1.0") -> None:
        self.name = name
        self.version = version
        self._tools: dict[str, "BaseTool"] = {}

    def register_tool(self, tool: "BaseTool") -> "MCPServer":
        self._tools[tool.name] = tool
        return self

    def _list_tools(self) -> list[MCPToolDescriptor]:
        return [
            MCPToolDescriptor(
                name=t.name,
                description=t.description,
                inputSchema=t.parameters,
            )
            for t in self._tools.values()
        ]

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Dispatch an MCP request and return a response."""
        try:
            if request.method == "initialize":
                return MCPResponse(
                    id=request.id,
                    result={
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": self.name, "version": self.version},
                        "capabilities": {"tools": {}},
                    },
                )

            elif request.method == "tools/list":
                tools = self._list_tools()
                return MCPResponse(
                    id=request.id,
                    result={"tools": [t.model_dump() for t in tools]},
                )

            elif request.method == "tools/call":
                tool_name = request.params.get("name", "")
                arguments = request.params.get("arguments", {})
                tool = self._tools.get(tool_name)
                if not tool:
                    return MCPResponse(
                        id=request.id,
                        error={"code": -32601, "message": f"Tool '{tool_name}' not found"},
                    )
                result = await tool.execute(arguments)
                content = [{"type": "text", "text": result.content}]
                if result.error:
                    content = [{"type": "text", "text": f"Error: {result.error}"}]
                return MCPResponse(
                    id=request.id,
                    result={"content": content, "isError": result.is_error},
                )

            else:
                return MCPResponse(
                    id=request.id,
                    error={"code": -32601, "message": f"Unknown method: {request.method}"},
                )

        except Exception as exc:
            return MCPResponse(
                id=request.id,
                error={"code": -32603, "message": str(exc)},
            )

    async def handle_json(self, raw: str) -> str:
        """Handle a raw JSON-RPC string and return JSON response."""
        try:
            data = json.loads(raw)
            request = MCPRequest(**data)
        except Exception as exc:
            return json.dumps(
                {"jsonrpc": "2.0", "error": {"code": -32700, "message": f"Parse error: {exc}"}}
            )
        response = await self.handle_request(request)
        return response.model_dump_json()

    async def serve_stdio(self) -> None:
        """Serve MCP over stdin/stdout (standard MCP transport)."""
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        loop = asyncio.get_event_loop()
        await loop.connect_read_pipe(lambda: protocol, __import__("sys").stdin)
        transport, _ = await loop.connect_write_pipe(
            asyncio.BaseProtocol, __import__("sys").stdout
        )

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break
                response_json = await self.handle_json(line.decode())
                transport.write((response_json + "\n").encode())
            except Exception:
                break
