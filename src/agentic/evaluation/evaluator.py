"""LLM-as-a-Judge evaluator and trajectory evaluation."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

from agentic.core.message import Message

if TYPE_CHECKING:
    from agentic.llm.base import BaseLLMProvider


class EvalScore(BaseModel):
    """Result of a single evaluation."""

    score: float  # 0.0 to 1.0
    reasoning: str
    criteria: str
    metadata: dict[str, Any] = {}


class EvalResult(BaseModel):
    """Aggregated evaluation of an agent response."""

    overall_score: float
    scores: dict[str, EvalScore] = {}
    passed: bool = True
    feedback: str = ""


JUDGE_PROMPT = """You are an expert evaluator. Rate the AI assistant's response on the following criterion.

Criterion: {criterion}
Description: {description}

User question: {question}
AI response: {response}
{reference}

Rate the response from 1 to 10 on this criterion.
Respond with:
Score: <number>
Reasoning: <brief explanation>"""


class LLMJudgeEvaluator:
    """Evaluates agent responses using an LLM as a judge.

    Supports multiple evaluation criteria (relevance, accuracy, completeness, etc.)

    Based on: MT-Bench (Zheng et al. 2023) and G-Eval (Liu et al. 2023).
    """

    DEFAULT_CRITERIA = {
        "relevance": "How relevant is the response to the question?",
        "accuracy": "How accurate and factually correct is the response?",
        "completeness": "How complete and thorough is the response?",
        "coherence": "How clear, coherent, and well-structured is the response?",
        "helpfulness": "How helpful is the response in addressing the user's needs?",
    }

    def __init__(
        self,
        judge_llm: "BaseLLMProvider",
        criteria: dict[str, str] | None = None,
        passing_threshold: float = 0.6,
    ) -> None:
        self.judge_llm = judge_llm
        self.criteria = criteria or self.DEFAULT_CRITERIA
        self.passing_threshold = passing_threshold

    async def _score_criterion(
        self,
        criterion: str,
        description: str,
        question: str,
        response: str,
        reference: str = "",
    ) -> EvalScore:
        ref_text = f"Reference answer: {reference}" if reference else ""
        prompt = JUDGE_PROMPT.format(
            criterion=criterion,
            description=description,
            question=question,
            response=response,
            reference=ref_text,
        )
        judge_response = await self.judge_llm.generate([Message.human(prompt)])
        raw = judge_response.message.content.strip()

        score = 5.0  # Default
        reasoning = raw
        for line in raw.split("\n"):
            if line.lower().startswith("score:"):
                try:
                    score = float(line.split(":")[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            elif line.lower().startswith("reasoning:"):
                reasoning = line.split(":", 1)[1].strip()

        return EvalScore(
            score=score / 10.0,
            reasoning=reasoning,
            criteria=criterion,
        )

    async def evaluate(
        self,
        question: str,
        response: str,
        reference: str = "",
        criteria: list[str] | None = None,
    ) -> EvalResult:
        """Evaluate a response across all (or specified) criteria."""
        import asyncio

        active_criteria = {
            k: v for k, v in self.criteria.items()
            if criteria is None or k in criteria
        }

        tasks = [
            self._score_criterion(k, v, question, response, reference)
            for k, v in active_criteria.items()
        ]
        scores_list = await asyncio.gather(*tasks, return_exceptions=True)

        scores: dict[str, EvalScore] = {}
        for (criterion, _), score_or_exc in zip(active_criteria.items(), scores_list):
            if isinstance(score_or_exc, Exception):
                scores[criterion] = EvalScore(
                    score=0.5, reasoning=f"Evaluation failed: {score_or_exc}", criteria=criterion
                )
            else:
                scores[criterion] = score_or_exc  # type: ignore[assignment]

        overall = sum(s.score for s in scores.values()) / len(scores) if scores else 0.0
        passed = overall >= self.passing_threshold

        feedback_parts = [
            f"{k}: {v.score:.1f}/1.0 — {v.reasoning}" for k, v in scores.items()
        ]
        feedback = "\n".join(feedback_parts)

        return EvalResult(
            overall_score=overall,
            scores=scores,
            passed=passed,
            feedback=feedback,
        )


class TrajectoryEvaluator:
    """Evaluates agent trajectories (sequences of tool calls and reasoning steps).

    Checks:
    - Whether the agent used tools appropriately
    - Whether the reasoning chain is sound
    - Whether the agent reached the goal efficiently
    """

    def __init__(self, judge_llm: "BaseLLMProvider") -> None:
        self.judge_llm = judge_llm

    async def evaluate_trajectory(
        self,
        goal: str,
        steps: list[str],
        final_answer: str,
    ) -> EvalResult:
        trajectory_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
        prompt = (
            f"Evaluate this agent's trajectory for achieving the goal.\n\n"
            f"Goal: {goal}\n\n"
            f"Steps taken:\n{trajectory_text}\n\n"
            f"Final answer: {final_answer}\n\n"
            f"Evaluate on:\n"
            f"1. Goal achievement (0-10)\n"
            f"2. Efficiency (0-10, fewer unnecessary steps = higher)\n"
            f"3. Reasoning quality (0-10)\n\n"
            f"Respond with:\n"
            f"Goal achievement: <score>\n"
            f"Efficiency: <score>\n"
            f"Reasoning quality: <score>\n"
            f"Feedback: <brief overall feedback>"
        )
        response = await self.judge_llm.generate([Message.human(prompt)])
        raw = response.message.content.strip()

        scores: dict[str, EvalScore] = {}
        feedback = ""
        for line in raw.split("\n"):
            for criterion in ["goal achievement", "efficiency", "reasoning quality"]:
                if line.lower().startswith(criterion + ":"):
                    try:
                        val = float(line.split(":")[1].strip().split()[0])
                        scores[criterion] = EvalScore(
                            score=val / 10.0, reasoning="", criteria=criterion
                        )
                    except (ValueError, IndexError):
                        pass
            if line.lower().startswith("feedback:"):
                feedback = line.split(":", 1)[1].strip()

        overall = sum(s.score for s in scores.values()) / len(scores) if scores else 0.5
        return EvalResult(
            overall_score=overall,
            scores=scores,
            passed=overall >= 0.6,
            feedback=feedback,
        )
