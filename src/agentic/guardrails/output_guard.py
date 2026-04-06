"""Output guardrails — filter and validate LLM responses."""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from agentic.exceptions import OutputGuardrailError
from agentic.guardrails.base import BaseGuardrail

if TYPE_CHECKING:
    from agentic.core.context import AgentContext
    from agentic.llm.base import BaseLLMProvider


# Basic toxic/harmful content patterns (heuristic; use a proper classifier in production)
_TOXIC_PATTERNS = [
    r"\b(?:kill|murder|harm|hurt|attack)\s+(?:yourself|themselves|people|humans)\b",
    r"\bhow\s+to\s+make\s+(?:a\s+)?(?:bomb|explosive|weapon|poison)\b",
    r"\bsynthes(?:ize|is)\s+(?:methamphetamine|cocaine|heroin|fentanyl)\b",
]
_COMPILED_TOXIC = [re.compile(p, re.IGNORECASE) for p in _TOXIC_PATTERNS]


class ToxicityFilter(BaseGuardrail):
    """Blocks or redacts outputs containing harmful content patterns."""

    def __init__(self, raise_on_detect: bool = True) -> None:
        self.raise_on_detect = raise_on_detect

    async def check_output(self, text: str, context: "AgentContext | None" = None) -> str:
        for pattern in _COMPILED_TOXIC:
            if pattern.search(text):
                if self.raise_on_detect:
                    raise OutputGuardrailError(
                        "Output blocked: contains potentially harmful content."
                    )
                return "[Response blocked by safety filter]"
        return text


class OutputLengthGuard(BaseGuardrail):
    """Enforces maximum output length (truncates if exceeded)."""

    def __init__(self, max_length: int = 10000) -> None:
        self.max_length = max_length

    async def check_output(self, text: str, context: "AgentContext | None" = None) -> str:
        if len(text) > self.max_length:
            return text[: self.max_length] + "\n\n[Output truncated by length limit]"
        return text


class LLMJudgeGuard(BaseGuardrail):
    """Uses an LLM to evaluate output quality/safety before returning to the user.

    Only blocks if the judge explicitly flags the response as harmful.
    """

    def __init__(
        self,
        judge_llm: "BaseLLMProvider",
        threshold_score: int = 3,
    ) -> None:
        self.judge_llm = judge_llm
        self.threshold_score = threshold_score

    async def check_output(self, text: str, context: "AgentContext | None" = None) -> str:
        from agentic.core.message import Message

        prompt = (
            f"Rate the following AI response for safety and appropriateness on a scale of 1-10.\n"
            f"1 = extremely harmful/unsafe, 10 = completely safe and appropriate.\n"
            f"Respond with ONLY a number.\n\nResponse:\n{text}\n\nSafety score:"
        )
        try:
            response = await self.judge_llm.generate([Message.human(prompt)])
            score_str = response.message.content.strip().split()[0]
            score = int(float(score_str))
            if score < self.threshold_score:
                raise OutputGuardrailError(
                    f"Output blocked by LLM judge (score {score}/{self.threshold_score})."
                )
        except OutputGuardrailError:
            raise
        except Exception:
            pass  # Don't block on judge errors
        return text


class FormatValidator(BaseGuardrail):
    """Validates that the output matches a required format (e.g., JSON, regex)."""

    def __init__(
        self,
        format: str = "any",  # "json", "regex", "any"
        regex_pattern: str = "",
        raise_on_invalid: bool = False,
    ) -> None:
        self.format = format
        self.regex_pattern = regex_pattern
        self._compiled = re.compile(regex_pattern) if regex_pattern else None
        self.raise_on_invalid = raise_on_invalid

    async def check_output(self, text: str, context: "AgentContext | None" = None) -> str:
        if self.format == "json":
            import json
            try:
                json.loads(text)
            except json.JSONDecodeError:
                if self.raise_on_invalid:
                    raise OutputGuardrailError("Output is not valid JSON.")
        elif self.format == "regex" and self._compiled:
            if not self._compiled.search(text):
                if self.raise_on_invalid:
                    raise OutputGuardrailError(
                        f"Output does not match required pattern: {self.regex_pattern}"
                    )
        return text
