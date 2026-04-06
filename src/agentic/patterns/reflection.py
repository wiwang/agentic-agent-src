"""Reflection (Producer-Critic) pattern — Chapter 4.

Implements iterative self-improvement where a generator produces content
and a critic evaluates it, looping until quality criteria are met.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

from agentic.core.message import Message
from agentic.exceptions import ReflectionError
from agentic.patterns.base import BasePattern, PatternResult

if TYPE_CHECKING:
    from agentic.core.context import AgentContext
    from agentic.llm.base import BaseLLMProvider


PRODUCER_SYSTEM = "You are an expert content creator. Produce high-quality, accurate, and detailed responses."

CRITIC_PROMPT = """You are a critical evaluator. Review the following response and provide structured feedback.

Original request: {request}

Response to evaluate:
{response}

Evaluate on:
1. Accuracy and correctness
2. Completeness
3. Clarity and coherence
4. Relevance to the request

If the response is satisfactory (score >= {threshold}/10), respond with: "APPROVED: <brief reason>"
Otherwise, respond with: "IMPROVE: <specific actionable feedback>"

Your evaluation:"""

REVISION_PROMPT = """Improve the following response based on the critic's feedback.

Original request: {request}

Current response:
{response}

Critic's feedback:
{feedback}

Improved response:"""


class ReflectionPattern(BasePattern):
    """Producer-Critic loop for iterative response improvement.

    Args:
        producer_llm: LLM used for generating and revising content.
        critic_llm: LLM used for evaluation (can be same as producer).
        max_iterations: Maximum critic-revision cycles.
        approval_threshold: Score (out of 10) at which critic approves.
        producer_system: System prompt for the producer.
    """

    def __init__(
        self,
        producer_llm: "BaseLLMProvider",
        critic_llm: "BaseLLMProvider | None" = None,
        max_iterations: int = 3,
        approval_threshold: int = 8,
        producer_system: str = PRODUCER_SYSTEM,
    ) -> None:
        self.producer_llm = producer_llm
        self.critic_llm = critic_llm or producer_llm
        self.max_iterations = max_iterations
        self.approval_threshold = approval_threshold
        self.producer_system = producer_system

    async def _produce(self, request: str, previous: str = "", feedback: str = "") -> str:
        if previous and feedback:
            prompt = REVISION_PROMPT.format(
                request=request, response=previous, feedback=feedback
            )
        else:
            prompt = request

        messages = [
            Message.system(self.producer_system),
            Message.human(prompt),
        ]
        response = await self.producer_llm.generate(messages)
        return response.message.content

    async def _critique(self, request: str, response: str) -> tuple[bool, str]:
        """Returns (approved, feedback)."""
        prompt = CRITIC_PROMPT.format(
            request=request,
            response=response,
            threshold=self.approval_threshold,
        )
        critic_response = await self.critic_llm.generate([Message.human(prompt)])
        feedback = critic_response.message.content.strip()
        approved = feedback.upper().startswith("APPROVED")
        return approved, feedback

    async def run(
        self,
        input: str,
        context: "AgentContext | None" = None,
        **kwargs: Any,
    ) -> PatternResult:
        steps = []
        current = ""
        feedback = ""

        for iteration in range(self.max_iterations):
            # Producer step
            current = await self._produce(input, previous=current, feedback=feedback)
            steps.append(f"[Iteration {iteration+1}] Producer: {current[:200]}...")

            # Critic step
            approved, feedback = await self._critique(input, current)
            steps.append(f"[Iteration {iteration+1}] Critic: {feedback[:200]}...")

            if approved:
                return PatternResult(
                    output=current,
                    steps=steps,
                    metadata={
                        "iterations": iteration + 1,
                        "approved": True,
                        "final_feedback": feedback,
                    },
                )

        # Return best attempt even if not formally approved
        return PatternResult(
            output=current,
            steps=steps,
            metadata={
                "iterations": self.max_iterations,
                "approved": False,
                "final_feedback": feedback,
            },
        )
