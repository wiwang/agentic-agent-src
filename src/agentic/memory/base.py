"""Abstract base class for memory systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


class BaseMemory(ABC):
    """Abstract memory interface.

    Memory in the agentic framework is responsible for:
    1. Storing interaction history (``store``)
    2. Retrieving relevant context for a query (``retrieve``)
    3. Clearing / resetting state (``clear``)
    """

    @abstractmethod
    async def store(
        self,
        input: str,
        output: str,
        context: "AgentContext | None" = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist an interaction for future retrieval."""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Return the most relevant memory strings for the query."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored memories."""

    async def get_all(self) -> list[dict[str, Any]]:
        """Return all stored memories as raw dicts (optional)."""
        return []
