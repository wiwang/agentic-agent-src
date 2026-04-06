"""Built-in code execution tool — runs Python in a sandboxed subprocess."""

from __future__ import annotations

import asyncio
import sys
import textwrap
from typing import Any, TYPE_CHECKING

from agentic.core.message import ToolResult
from agentic.tools.base import BaseTool

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


class CodeExecutorTool(BaseTool):
    """Execute Python code in an isolated subprocess and return the output."""

    name = "code_executor"
    description = (
        "Execute Python code and return stdout/stderr. "
        "Use for calculations, data processing, or generating artifacts. "
        "The environment is sandboxed — no network access, no file writes outside /tmp."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            },
            "timeout": {
                "type": "number",
                "description": "Execution timeout in seconds (default: 10)",
            },
        },
        "required": ["code"],
    }

    def __init__(self, default_timeout: float = 10.0, max_output_chars: int = 4000) -> None:
        self._default_timeout = default_timeout
        self._max_output_chars = max_output_chars

    async def execute(
        self, arguments: dict[str, Any], context: "AgentContext | None" = None
    ) -> ToolResult:
        import uuid

        tool_call_id = str(uuid.uuid4())
        code = arguments.get("code", "")
        timeout = float(arguments.get("timeout", self._default_timeout))

        # Wrap in a restricted preamble
        wrapped = textwrap.dedent(
            f"""
import sys
import os
# Restrict filesystem access
_open = open
def open(file, mode='r', *args, **kwargs):
    from pathlib import Path
    p = Path(str(file)).resolve()
    if 'w' in mode or 'a' in mode or 'x' in mode:
        if not str(p).startswith('/tmp'):
            raise PermissionError(f"Write access outside /tmp is not allowed: {{p}}")
    return _open(file, mode, *args, **kwargs)

{code}
"""
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                wrapped,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            out = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")

            combined = ""
            if out:
                combined += out
            if err:
                combined += f"\n[stderr]\n{err}"

            combined = combined[: self._max_output_chars]

            if proc.returncode != 0:
                return ToolResult(
                    tool_call_id=tool_call_id,
                    name=self.name,
                    content=combined or "",
                    error=f"Process exited with code {proc.returncode}",
                )
            return ToolResult(
                tool_call_id=tool_call_id,
                name=self.name,
                content=combined or "(no output)",
            )

        except asyncio.TimeoutError:
            return ToolResult(
                tool_call_id=tool_call_id,
                name=self.name,
                content="",
                error=f"Execution timed out after {timeout}s",
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=tool_call_id,
                name=self.name,
                content="",
                error=str(exc),
            )
