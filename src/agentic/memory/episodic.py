"""Episodic memory — stores and retrieves structured experience episodes."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field

from agentic.memory.base import BaseMemory

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


class Episode(BaseModel):
    """A single episodic memory unit."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: str
    approach: str
    outcome: str
    success: bool
    tags: list[str] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodicMemory(BaseMemory):
    """Stores structured episodes (task → approach → outcome) and retrieves
    similar past experiences using keyword overlap.

    For richer retrieval, combine with VectorStoreMemory.
    """

    def __init__(
        self,
        persist_path: str | None = None,
        max_episodes: int = 500,
    ) -> None:
        self._persist_path = Path(persist_path) if persist_path else None
        self._max_episodes = max_episodes
        self._episodes: list[Episode] = []
        if self._persist_path and self._persist_path.exists():
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._persist_path.read_text())  # type: ignore[union-attr]
            self._episodes = [Episode(**ep) for ep in data]
        except Exception:
            self._episodes = []

    def _save(self) -> None:
        if self._persist_path:
            self._persist_path.write_text(
                json.dumps([ep.model_dump() for ep in self._episodes], indent=2)
            )

    async def add_episode(
        self,
        task: str,
        approach: str,
        outcome: str,
        success: bool,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        ep = Episode(
            task=task,
            approach=approach,
            outcome=outcome,
            success=success,
            tags=tags or [],
            metadata=metadata or {},
        )
        self._episodes.append(ep)
        # Trim oldest if over limit
        if len(self._episodes) > self._max_episodes:
            self._episodes = self._episodes[-self._max_episodes:]
        self._save()
        return ep

    async def store(
        self,
        input: str,
        output: str,
        context: "AgentContext | None" = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Auto-store interaction as an unstructured episode."""
        await self.add_episode(
            task=input,
            approach="agent_response",
            outcome=output,
            success=True,
            metadata=metadata or {},
        )

    async def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve episodes most relevant to query using keyword overlap."""
        if not self._episodes:
            return []
        query_words = set(query.lower().split())
        scored: list[tuple[float, Episode]] = []
        for ep in self._episodes:
            ep_text = f"{ep.task} {ep.approach} {ep.outcome}".lower()
            ep_words = set(ep_text.split())
            overlap = len(query_words & ep_words)
            if overlap > 0:
                scored.append((overlap, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, ep in scored[:top_k]:
            status = "✓" if ep.success else "✗"
            results.append(
                f"[{status}] Task: {ep.task}\nApproach: {ep.approach}\nOutcome: {ep.outcome}"
            )
        return results

    async def clear(self) -> None:
        self._episodes.clear()
        self._save()

    async def get_all(self) -> list[dict[str, Any]]:
        return [ep.model_dump() for ep in self._episodes]

    def successful_episodes(self) -> list[Episode]:
        return [ep for ep in self._episodes if ep.success]

    def failed_episodes(self) -> list[Episode]:
        return [ep for ep in self._episodes if not ep.success]
