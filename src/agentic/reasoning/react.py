"""ReAct (Reason + Act) reasoning strategy.

Implements the Thought → Action → Observation loop from Yao et al. (2022).
The agent interleaves reasoning traces with tool actions.
"""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from agentic.core.message import Message, ToolCall, ToolResult
from agentic.reasoning.base import BaseReasoner, ReasoningResult

if TYPE_CHECKING:
    from agentic.core.context import AgentContext
    from agentic.llm.base import BaseLLMProvider
    from agentic.tools.base import BaseTool


REACT_SYSTEM_PROMPT = """You are a ReAct agent that solves problems by alternating between Thought, Action, and Observation.

Format your responses EXACTLY as:
Thought: <your reasoning about what to do next>
Action: <tool_name>(<arg1>=<val1>, <arg2>=<val2>)
OR if you have the final answer:
Thought: <your final reasoning>
Answer: <your final answer>

Available tools:
{tools}

Important: Only use the tools listed above. Always start with Thought:"""


class ReActReasoner(BaseReasoner):
    """ReAct reasoner: Thought → Action → Observation loop.

    Integrates with BaseTool instances for executing actions.
    """

    def __init__(
        self,
        llm: "BaseLLMProvider",
        tools: list["BaseTool"] | None = None,
        max_steps: int = 10,
    ) -> None:
        super().__init__(llm)
        self._tools: dict[str, "BaseTool"] = {t.name: t for t in (tools or [])}
        self.max_steps = max_steps

    def add_tool(self, tool: "BaseTool") -> None:
        self._tools[tool.name] = tool

    def _format_tools(self) -> str:
        if not self._tools:
            return "No tools available."
        lines = []
        for t in self._tools.values():
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)

    def _parse_action(self, text: str) -> tuple[str, dict[str, Any]] | None:
        """Parse 'tool_name(arg=val, ...)' into (name, args)."""
        match = re.match(r"(\w+)\((.*)\)$", text.strip(), re.DOTALL)
        if not match:
            return None
        tool_name = match.group(1)
        args_str = match.group(2)
        args: dict[str, Any] = {}
        # Parse simple key=value pairs
        for part in re.split(r",\s*(?=\w+=)", args_str):
            kv = part.split("=", 1)
            if len(kv) == 2:
                key = kv[0].strip()
                val = kv[1].strip().strip("'\"")
                args[key] = val
        return tool_name, args

    async def reason(
        self,
        question: str,
        context: "AgentContext | None" = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        system = REACT_SYSTEM_PROMPT.format(tools=self._format_tools())
        messages = [
            Message.system(system),
            Message.human(question),
        ]
        trace: list[str] = []

        for step in range(self.max_steps):
            response = await self.llm.generate(messages)
            raw = response.message.content.strip()
            messages.append(response.message)
            trace.append(raw)

            # Check for final answer
            answer_match = re.search(r"(?i)answer:\s*(.+)", raw, re.DOTALL)
            if answer_match:
                return ReasoningResult(
                    answer=answer_match.group(1).strip(),
                    reasoning_trace=trace,
                    metadata={"steps": step + 1},
                )

            # Parse action
            action_match = re.search(r"(?i)action:\s*(.+?)(?:\n|$)", raw)
            if action_match:
                action_text = action_match.group(1).strip()
                parsed = self._parse_action(action_text)

                if parsed:
                    tool_name, args = parsed
                    tool = self._tools.get(tool_name)
                    if tool:
                        result = await tool.execute(args, context=context)
                        observation = result.content if not result.error else f"Error: {result.error}"
                    else:
                        observation = f"Tool '{tool_name}' not found. Available: {list(self._tools.keys())}"
                else:
                    observation = f"Could not parse action: {action_text}"

                obs_msg = f"Observation: {observation}"
                trace.append(obs_msg)
                messages.append(Message.human(obs_msg))
            else:
                # No action and no answer — treat raw as answer
                return ReasoningResult(
                    answer=raw,
                    reasoning_trace=trace,
                    metadata={"steps": step + 1, "early_stop": True},
                )

        return ReasoningResult(
            answer="Max steps reached without a final answer.",
            reasoning_trace=trace,
            confidence=0.0,
            metadata={"steps": self.max_steps, "max_steps_reached": True},
        )
