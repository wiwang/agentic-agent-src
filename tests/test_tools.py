"""Tests for the tool system."""

from __future__ import annotations

import pytest

from agentic.tools.base import BaseTool, FunctionTool, tool
from agentic.tools.registry import ToolRegistry
from agentic.tools.builtin.calculator import CalculatorTool
from agentic.tools.builtin.code_executor import CodeExecutorTool
from agentic.core.message import ToolResult


# ── CalculatorTool ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_calculator_basic() -> None:
    calc = CalculatorTool()
    result = await calc.execute({"expression": "2 + 2"})
    assert result.content == "4"
    assert not result.is_error


@pytest.mark.asyncio
async def test_calculator_complex() -> None:
    calc = CalculatorTool()
    result = await calc.execute({"expression": "sqrt(144) + 2**10"})
    assert result.content == "1036"


@pytest.mark.asyncio
async def test_calculator_invalid() -> None:
    calc = CalculatorTool()
    result = await calc.execute({"expression": "import os; os.system('rm -rf /')"})
    assert result.is_error


@pytest.mark.asyncio
async def test_calculator_float_result() -> None:
    calc = CalculatorTool()
    result = await calc.execute({"expression": "1 / 3"})
    assert "0.333" in result.content


# ── CodeExecutorTool ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_code_executor_basic() -> None:
    executor = CodeExecutorTool()
    result = await executor.execute({"code": "print('hello world')"})
    assert "hello world" in result.content
    assert not result.is_error


@pytest.mark.asyncio
async def test_code_executor_timeout() -> None:
    executor = CodeExecutorTool(default_timeout=0.5)
    result = await executor.execute({"code": "import time; time.sleep(5)", "timeout": 0.5})
    assert result.is_error
    assert "timed out" in result.error.lower()


@pytest.mark.asyncio
async def test_code_executor_with_output() -> None:
    executor = CodeExecutorTool()
    result = await executor.execute({"code": "for i in range(5): print(i)"})
    assert "0" in result.content
    assert "4" in result.content


# ── @tool decorator ───────────────────────────────────────────────────────────

def test_tool_decorator_sync() -> None:
    @tool(name="greet", description="Greet someone")
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    assert isinstance(greet, FunctionTool)
    assert greet.name == "greet"
    assert greet.description == "Greet someone"
    assert "name" in greet.parameters["properties"]


@pytest.mark.asyncio
async def test_tool_decorator_execution() -> None:
    @tool(description="Add two numbers")
    async def add(a: int, b: int) -> int:
        return a + b

    result = await add.execute({"a": 3, "b": 4})
    assert result.content == "7"


@pytest.mark.asyncio
async def test_tool_decorator_error_handling() -> None:
    @tool(description="Always fails")
    def failing_tool(x: str) -> str:
        raise ValueError("Intentional error")

    result = await failing_tool.execute({"x": "test"})
    assert result.is_error
    assert "Intentional error" in result.error


# ── ToolRegistry ──────────────────────────────────────────────────────────────

def test_registry_register_and_get() -> None:
    registry = ToolRegistry()
    calc = CalculatorTool()
    registry.register(calc)
    assert "calculator" in registry
    assert registry.get("calculator") is calc


def test_registry_not_found() -> None:
    from agentic.exceptions import ToolNotFoundError
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        registry.get("nonexistent")


def test_registry_schemas() -> None:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    schemas = registry.schemas()
    assert len(schemas) == 1
    assert schemas[0].name == "calculator"
