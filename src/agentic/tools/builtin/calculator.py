"""Built-in Calculator tool — evaluates safe mathematical expressions."""

from __future__ import annotations

import ast
import math
import operator as op
from typing import Any, TYPE_CHECKING

from agentic.core.message import ToolResult
from agentic.tools.base import BaseTool

if TYPE_CHECKING:
    from agentic.core.context import AgentContext

# Allowed operators for safe evaluation
_OPERATORS: dict[type, Any] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.BitXor: op.xor,
    ast.USub: op.neg,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}

_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_func = _OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_func = _OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported function call")
        func = _SAFE_FUNCTIONS.get(node.func.id)
        if func is None:
            raise ValueError(f"Unknown function: {node.func.id}")
        args = [_safe_eval(a) for a in node.args]
        return func(*args)  # type: ignore[operator]
    if isinstance(node, ast.Name):
        val = _SAFE_FUNCTIONS.get(node.id)
        if val is None or not isinstance(val, float):
            raise ValueError(f"Unknown name: {node.id}")
        return val
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


class CalculatorTool(BaseTool):
    """Safely evaluate mathematical expressions."""

    name = "calculator"
    description = (
        "Evaluate a mathematical expression and return the numeric result. "
        "Supports +, -, *, /, **, %, sqrt, sin, cos, tan, log, exp, abs, round, pi, e."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A mathematical expression, e.g. '2 ** 10 + sqrt(16)'",
            }
        },
        "required": ["expression"],
    }

    async def execute(
        self, arguments: dict[str, Any], context: "AgentContext | None" = None
    ) -> ToolResult:
        import uuid

        tool_call_id = str(uuid.uuid4())
        expression = arguments.get("expression", "")
        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree.body)
            # Format nicely: integer if result is a whole number
            if result == int(result):
                content = str(int(result))
            else:
                content = str(result)
            return ToolResult(tool_call_id=tool_call_id, name=self.name, content=content)
        except Exception as exc:
            return ToolResult(
                tool_call_id=tool_call_id,
                name=self.name,
                content="",
                error=f"Calculation error: {exc}",
            )
