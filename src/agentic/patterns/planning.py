"""Planning pattern — Chapter 6.

Implements Plan-then-Execute: the LLM first generates a structured plan,
then executes each step using tools or sub-agents.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from pydantic import BaseModel

from agentic.core.message import Message
from agentic.exceptions import PlanningError
from agentic.patterns.base import BasePattern, PatternResult

if TYPE_CHECKING:
    from agentic.core.context import AgentContext
    from agentic.llm.base import BaseLLMProvider
    from agentic.tools.base import BaseTool


class PlanStep(BaseModel):
    """A single step in an agent plan."""

    id: int
    action: str
    tool: str | None = None
    tool_args: dict[str, Any] = {}
    depends_on: list[int] = []
    result: str = ""
    completed: bool = False


class Plan(BaseModel):
    """A structured multi-step execution plan."""

    goal: str
    steps: list[PlanStep]
    reasoning: str = ""


PLANNER_PROMPT = """You are a planning agent. Create a detailed step-by-step plan to achieve the goal.

Available tools: {tools}

Goal: {goal}

Respond with a JSON plan in this format:
{{
  "goal": "...",
  "reasoning": "Brief reasoning about the approach",
  "steps": [
    {{
      "id": 1,
      "action": "Description of what to do",
      "tool": "tool_name or null",
      "tool_args": {{"arg1": "value1"}},
      "depends_on": []
    }}
  ]
}}

Plan (JSON only, no markdown):"""

EXECUTOR_PROMPT = """Execute the following step as part of a larger plan.

Overall goal: {goal}
Current step: {action}
Context from previous steps:
{context}

Execute this step and provide the result:"""


class PlanningPattern(BasePattern):
    """Plan-then-Execute pattern.

    1. Use the LLM to generate a structured plan with steps.
    2. Execute each step sequentially (or in parallel if no dependencies).
    3. Return the final aggregated result.
    """

    def __init__(
        self,
        llm: "BaseLLMProvider",
        tools: list["BaseTool"] | None = None,
        replan_on_failure: bool = True,
    ) -> None:
        self.llm = llm
        self._tools: dict[str, "BaseTool"] = {t.name: t for t in (tools or [])}
        self.replan_on_failure = replan_on_failure

    def add_tool(self, tool: "BaseTool") -> None:
        self._tools[tool.name] = tool

    async def _generate_plan(self, goal: str) -> Plan:
        tool_list = ", ".join(self._tools.keys()) or "none"
        prompt = PLANNER_PROMPT.format(tools=tool_list, goal=goal)
        response = await self.llm.generate([Message.human(prompt)])

        # Extract JSON
        text = response.message.content.strip()
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

        try:
            data = json.loads(text)
            steps = [PlanStep(**s) for s in data.get("steps", [])]
            return Plan(
                goal=data.get("goal", goal),
                steps=steps,
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, TypeError) as e:
            raise PlanningError(f"Failed to parse plan: {e}\nRaw: {text}") from e

    async def _execute_step(
        self,
        step: PlanStep,
        plan: Plan,
        results: dict[int, str],
        context: Any,
    ) -> str:
        context_str = "\n".join(
            f"Step {sid}: {results[sid]}" for sid in step.depends_on if sid in results
        )

        # Use tool if specified
        if step.tool and step.tool in self._tools:
            tool = self._tools[step.tool]
            result = await tool.execute(step.tool_args, context=context)
            return result.content if not result.error else f"Error: {result.error}"

        # Use LLM to execute
        prompt = EXECUTOR_PROMPT.format(
            goal=plan.goal, action=step.action, context=context_str or "None"
        )
        response = await self.llm.generate([Message.human(prompt)])
        return response.message.content

    async def run(
        self,
        input: str,
        context: "AgentContext | None" = None,
        **kwargs: Any,
    ) -> PatternResult:
        plan = await self._generate_plan(input)
        results: dict[int, str] = {}
        steps_log: list[str] = [f"Plan: {plan.reasoning}"]

        for step in plan.steps:
            try:
                result = await self._execute_step(step, plan, results, context)
                results[step.id] = result
                step.result = result
                step.completed = True
                steps_log.append(f"Step {step.id} ({step.action}): {result[:150]}...")
            except Exception as exc:
                results[step.id] = f"[Failed: {exc}]"
                steps_log.append(f"Step {step.id} FAILED: {exc}")

        # Aggregate final result
        final_parts = [f"Step {s.id}: {s.action}\nResult: {results.get(s.id, 'N/A')}" for s in plan.steps]
        summary_prompt = (
            f"Based on the following executed plan for '{input}', "
            f"provide a final comprehensive answer:\n\n" + "\n\n".join(final_parts)
        )
        final_response = await self.llm.generate([Message.human(summary_prompt)])

        return PatternResult(
            output=final_response.message.content,
            steps=steps_log,
            metadata={
                "plan_steps": len(plan.steps),
                "completed_steps": sum(1 for s in plan.steps if s.completed),
                "plan_reasoning": plan.reasoning,
            },
        )
