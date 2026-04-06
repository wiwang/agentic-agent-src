"""Short-term (in-memory) conversation buffer memory."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from agentic.memory.base import BaseMemory

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


class ConversationBufferMemory(BaseMemory):
    """Keeps the last N conversation turns in memory (FIFO ring buffer).

    This is the simplest form of short-term memory — it stores raw text pairs
    and returns the most recent ones as formatted context strings.
    """

    def __init__(self, max_turns: int = 20) -> None:
        self._max_turns = max_turns
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_turns)

    async def store(
        self,
        input: str,
        output: str,
        context: "AgentContext | None" = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._buffer.append(
            {
                "input": input,
                "output": output,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }
        )

    async def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Return the most recent turns as formatted strings (no semantic search)."""
        recent = list(self._buffer)[-top_k:]
        result = []
        for turn in recent:
            result.append(f"User: {turn['input']}\nAssistant: {turn['output']}")
        return result

    async def clear(self) -> None:
        self._buffer.clear()

    async def get_all(self) -> list[dict[str, Any]]:
        return list(self._buffer)

    @property
    def turn_count(self) -> int:
        return len(self._buffer)


class SlidingWindowMemory(BaseMemory):
    """Token-aware sliding-window memory that fits within a token budget."""

    def __init__(self, max_tokens: int = 2000, chars_per_token: float = 4.0) -> None:
        self._max_chars = int(max_tokens * chars_per_token)
        self._turns: list[dict[str, Any]] = []

    async def store(
        self,
        input: str,
        output: str,
        context: "AgentContext | None" = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._turns.append(
            {
                "input": input,
                "output": output,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        # Trim oldest until we're within budget
        self._trim()

    def _trim(self) -> None:
        total = sum(
            len(t["input"]) + len(t["output"]) for t in self._turns
        )
        while total > self._max_chars and self._turns:
            removed = self._turns.pop(0)
            total -= len(removed["input"]) + len(removed["output"])

    async def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        recent = self._turns[-top_k:]
        return [f"User: {t['input']}\nAssistant: {t['output']}" for t in recent]

    async def clear(self) -> None:
        self._turns.clear()
