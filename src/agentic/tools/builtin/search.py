"""Built-in web search tool using DuckDuckGo (no API key required)."""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from agentic.core.message import ToolResult
from agentic.tools.base import BaseTool

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


class WebSearchTool(BaseTool):
    """Search the web using DuckDuckGo Instant Answer API."""

    name = "web_search"
    description = (
        "Search the web for information. Returns a list of search results with "
        "titles, URLs, and snippets."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5)",
            },
        },
        "required": ["query"],
    }

    def __init__(self, max_results: int = 5, timeout: float = 10.0) -> None:
        self._default_max = max_results
        self._timeout = timeout

    async def execute(
        self, arguments: dict[str, Any], context: "AgentContext | None" = None
    ) -> ToolResult:
        import uuid
        import httpx

        tool_call_id = str(uuid.uuid4())
        query = arguments.get("query", "")
        max_results = int(arguments.get("max_results", self._default_max))

        try:
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            results = []
            # Abstract (main answer)
            if data.get("Abstract"):
                results.append(
                    {
                        "title": data.get("Heading", "Result"),
                        "url": data.get("AbstractURL", ""),
                        "snippet": data["Abstract"],
                    }
                )

            # Related topics
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append(
                        {
                            "title": topic.get("Text", "")[:80],
                            "url": topic.get("FirstURL", ""),
                            "snippet": topic.get("Text", ""),
                        }
                    )

            if not results:
                content = f"No results found for: {query}"
            else:
                content = json.dumps(results[:max_results], ensure_ascii=False, indent=2)

            return ToolResult(tool_call_id=tool_call_id, name=self.name, content=content)

        except Exception as exc:
            return ToolResult(
                tool_call_id=tool_call_id,
                name=self.name,
                content="",
                error=f"Search failed: {exc}",
            )
