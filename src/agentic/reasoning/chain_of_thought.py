"""Chain-of-Thought (CoT) reasoning strategy.

Implements both zero-shot CoT ("Let's think step by step") and
few-shot CoT (with example reasoning chains).
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from agentic.core.message import Message
from agentic.reasoning.base import BaseReasoner, ReasoningResult

if TYPE_CHECKING:
    from agentic.core.context import AgentContext
    from agentic.llm.base import BaseLLMProvider


ZERO_SHOT_COT_PROMPT = """You are a careful reasoner. Think step by step before giving your final answer.

Question: {question}

Let's think step by step:"""

FEW_SHOT_COT_TEMPLATE = """You are a careful reasoner. Here are some examples of step-by-step reasoning:

{examples}

Now answer the following question using the same step-by-step approach:

Question: {question}

Let's think step by step:"""


class ChainOfThoughtReasoner(BaseReasoner):
    """Applies Chain-of-Thought prompting to elicit systematic reasoning.

    Modes:
    - ``zero_shot``: Appends "Let's think step by step" (Kojima et al. 2022)
    - ``few_shot``: Provides example reasoning chains before the question
    """

    def __init__(
        self,
        llm: "BaseLLMProvider",
        mode: str = "zero_shot",
        examples: list[dict[str, str]] | None = None,
    ) -> None:
        """
        Args:
            llm: LLM provider to use for reasoning.
            mode: "zero_shot" or "few_shot".
            examples: List of {"question": ..., "reasoning": ..., "answer": ...} dicts.
        """
        super().__init__(llm)
        self.mode = mode
        self.examples = examples or []

    def _build_prompt(self, question: str) -> str:
        if self.mode == "few_shot" and self.examples:
            example_text = "\n\n".join(
                f"Question: {ex['question']}\nReasoning: {ex['reasoning']}\nAnswer: {ex['answer']}"
                for ex in self.examples
            )
            return FEW_SHOT_COT_TEMPLATE.format(
                examples=example_text, question=question
            )
        return ZERO_SHOT_COT_PROMPT.format(question=question)

    async def reason(
        self,
        question: str,
        context: "AgentContext | None" = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        prompt = self._build_prompt(question)
        messages = [Message.human(prompt)]

        response = await self.llm.generate(messages)
        raw = response.message.content

        # Split reasoning from answer (heuristic: last line starting with "Answer:" or "Therefore:")
        lines = raw.strip().split("\n")
        answer = raw
        reasoning_trace = []

        for i, line in enumerate(lines):
            low = line.strip().lower()
            if low.startswith(("answer:", "therefore:", "so the answer", "final answer")):
                reasoning_trace = lines[:i]
                answer = "\n".join(lines[i:]).strip()
                break
        else:
            # No explicit answer delimiter — use last paragraph
            paragraphs = raw.strip().split("\n\n")
            if len(paragraphs) > 1:
                reasoning_trace = paragraphs[:-1]
                answer = paragraphs[-1]

        return ReasoningResult(
            answer=answer,
            reasoning_trace=[r if isinstance(r, str) else "\n".join(r) for r in reasoning_trace],
            metadata={"mode": self.mode, "model": response.model},
        )
