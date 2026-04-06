"""Abstract base class for reasoning strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from agentic.core.context import AgentContext
    from agentic.llm.base import BaseLLMProvider


class ReasoningResult(BaseModel):
    """Output of a reasoning strategy."""

    answer: str
    reasoning_trace: list[str] = []
    confidence: float = 1.0
    metadata: dict[str, Any] = {}


class BaseReasoner(ABC):
    """Abstract interface for reasoning strategies.

    All reasoning strategies produce a ReasoningResult from a question + context.
    """

    def __init__(self, llm: "BaseLLMProvider") -> None:
        self.llm = llm

    @abstractmethod
    async def reason(
        self,
        question: str,
        context: "AgentContext | None" = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        """Apply the reasoning strategy to produce an answer."""
