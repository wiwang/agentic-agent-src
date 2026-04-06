"""Prompt Chaining pattern — Chapter 1.

Executes a sequence of LLM calls where each step's output feeds the next.
Supports conditional branching and output transformation between steps.
"""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from agentic.core.message import Message
from agentic.patterns.base import BasePattern, PatternResult

if TYPE_CHECKING:
    from agentic.core.context import AgentContext
    from agentic.llm.base import BaseLLMProvider


class ChainStep(BaseModel):
    """A single step in a prompt chain."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    prompt_template: str  # Use {input} and {output_N} placeholders
    system_prompt: str = ""
    transform: str | None = None  # Python expression for post-processing (safe eval)


class PromptChainingPattern(BasePattern):
    """Sequential prompt chaining: output of step N → input of step N+1.

    Steps can reference:
    - ``{input}``: the original user input
    - ``{previous}``: the immediately preceding step's output
    - ``{step_N}``: the output of step N (0-indexed)

    Example::

        chain = PromptChainingPattern(
            llm=llm,
            steps=[
                ChainStep(name="extract", prompt_template="Extract key facts from: {input}"),
                ChainStep(name="summarize", prompt_template="Summarize these facts: {previous}"),
                ChainStep(name="conclude", prompt_template="Draw conclusions from: {step_1}"),
            ]
        )
    """

    def __init__(
        self,
        llm: "BaseLLMProvider",
        steps: list[ChainStep] | None = None,
        stop_on_error: bool = True,
    ) -> None:
        self.llm = llm
        self.steps: list[ChainStep] = steps or []
        self.stop_on_error = stop_on_error

    def add_step(
        self,
        name: str,
        prompt_template: str,
        system_prompt: str = "",
    ) -> "PromptChainingPattern":
        self.steps.append(
            ChainStep(name=name, prompt_template=prompt_template, system_prompt=system_prompt)
        )
        return self

    def _render(self, template: str, input: str, outputs: list[str]) -> str:
        ctx: dict[str, Any] = {"input": input, "previous": outputs[-1] if outputs else input}
        for i, out in enumerate(outputs):
            ctx[f"step_{i}"] = out
        for k, v in ctx.items():
            template = template.replace("{" + k + "}", v)
        return template

    async def run(
        self,
        input: str,
        context: "AgentContext | None" = None,
        **kwargs: Any,
    ) -> PatternResult:
        outputs: list[str] = []
        steps_log: list[str] = []

        for i, step in enumerate(self.steps):
            prompt = self._render(step.prompt_template, input, outputs)
            messages = []
            if step.system_prompt:
                messages.append(Message.system(step.system_prompt))
            messages.append(Message.human(prompt))

            try:
                response = await self.llm.generate(messages)
                output = response.message.content
            except Exception as exc:
                if self.stop_on_error:
                    return PatternResult(
                        output="\n".join(outputs),
                        steps=steps_log,
                        success=False,
                        error=f"Step '{step.name}' failed: {exc}",
                    )
                output = f"[Step '{step.name}' failed: {exc}]"

            outputs.append(output)
            steps_log.append(f"[{step.name}]: {output[:200]}...")

        return PatternResult(
            output=outputs[-1] if outputs else "",
            steps=steps_log,
            metadata={
                "n_steps": len(self.steps),
                "all_outputs": outputs,
            },
        )
