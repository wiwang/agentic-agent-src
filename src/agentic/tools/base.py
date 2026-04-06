"""BaseTool and @tool decorator."""

from __future__ import annotations

import asyncio
import functools
import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, TYPE_CHECKING

from pydantic import BaseModel

from agentic.core.message import ToolResult
from agentic.llm.base import ToolSchema

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


class BaseTool(ABC):
    """Abstract base class for all tools.

    Subclasses must define:
        - ``name``
        - ``description``
        - ``parameters`` (JSON Schema dict)
        - ``execute(arguments, context) -> ToolResult``
    """

    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    async def execute(
        self, arguments: dict[str, Any], context: "AgentContext | None" = None
    ) -> ToolResult:
        """Execute the tool and return a ToolResult."""

    def to_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def __repr__(self) -> str:
        return f"<Tool name={self.name!r}>"


class FunctionTool(BaseTool):
    """Tool created from a regular or async Python function."""

    def __init__(
        self,
        func: Callable[..., Any],
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        self._func = func
        self.name = name
        self.description = description
        self.parameters = parameters

    async def execute(
        self, arguments: dict[str, Any], context: "AgentContext | None" = None
    ) -> ToolResult:
        import uuid

        tool_call_id = str(uuid.uuid4())
        try:
            # Support both sync and async functions
            if asyncio.iscoroutinefunction(self._func):
                result = await self._func(**arguments)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, functools.partial(self._func, **arguments)
                )
            content = str(result) if not isinstance(result, str) else result
            return ToolResult(tool_call_id=tool_call_id, name=self.name, content=content)
        except Exception as exc:
            return ToolResult(
                tool_call_id=tool_call_id,
                name=self.name,
                content="",
                error=str(exc),
            )


def _infer_json_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Infer a basic JSON Schema from a function's type annotations."""
    sig = inspect.signature(func)
    hints = {}
    try:
        hints = func.__annotations__
    except AttributeError:
        pass

    TYPE_MAP = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
    }

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "context"):
            continue
        annotation = hints.get(param_name, inspect.Parameter.empty)
        if annotation is inspect.Parameter.empty:
            json_type = "string"
        else:
            type_name = getattr(annotation, "__name__", str(annotation))
            json_type = TYPE_MAP.get(type_name, "string")

        properties[param_name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def tool(
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], FunctionTool]:
    """Decorator to turn a function into a FunctionTool.

    Usage::

        @tool(description="Search the web")
        async def web_search(query: str) -> str:
            ...
    """

    def decorator(func: Callable[..., Any]) -> FunctionTool:
        _name = name or func.__name__
        _description = description or (func.__doc__ or "").strip() or _name
        _parameters = parameters or _infer_json_schema(func)
        return FunctionTool(
            func=func,
            name=_name,
            description=_description,
            parameters=_parameters,
        )

    return decorator
