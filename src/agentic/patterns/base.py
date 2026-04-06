"""Abstract base class for all agentic design patterns."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


class PatternResult(BaseModel):
    """Result returned by any design pattern."""

    output: str
    metadata: dict[str, Any] = {}
    steps: list[str] = []
    success: bool = True
    error: str | None = None


class BasePattern(ABC):
    """Abstract interface for agentic design patterns.

    Every pattern transforms input → PatternResult, optionally using
    an LLM provider, tools, and shared context.
    """

    @abstractmethod
    async def run(
        self,
        input: str,
        context: "AgentContext | None" = None,
        **kwargs: Any,
    ) -> PatternResult:
        """Execute the pattern and return a result."""
