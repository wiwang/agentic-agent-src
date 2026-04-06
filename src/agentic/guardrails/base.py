"""Abstract base class for guardrails."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


class GuardrailResult(BaseModel):
    """Result of a guardrail check."""

    passed: bool
    text: str  # Possibly modified text
    reason: str = ""
    metadata: dict[str, Any] = {}


class BaseGuardrail(ABC):
    """Abstract interface for input and output guardrails.

    Subclasses may implement ``check_input``, ``check_output``, or both.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    async def check_input(self, text: str, context: "AgentContext | None" = None) -> str:
        """Validate/sanitize input. Raise InputGuardrailError to block."""
        return text

    async def check_output(self, text: str, context: "AgentContext | None" = None) -> str:
        """Filter/sanitize output. Raise OutputGuardrailError to block."""
        return text
