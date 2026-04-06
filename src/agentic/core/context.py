"""AgentContext — carries shared state through an agent's execution."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentic.core.message import Message


class AgentState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING_FOR_TOOL = "waiting_for_tool"
    WAITING_FOR_HUMAN = "waiting_for_human"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentContext(BaseModel):
    """Mutable execution context for a single agent run."""

    model_config = {"arbitrary_types_allowed": True}

    # Identity
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # State
    state: AgentState = AgentState.IDLE
    iteration: int = 0
    max_iterations: int = 20

    # Conversation history
    messages: list[Message] = Field(default_factory=list)

    # Key-value store for arbitrary state shared across patterns
    store: dict[str, Any] = Field(default_factory=dict)

    # Timing
    started_at: datetime | None = None
    ended_at: datetime | None = None

    # Token tracking
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    def start(self) -> None:
        self.state = AgentState.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def complete(self) -> None:
        self.state = AgentState.COMPLETED
        self.ended_at = datetime.now(timezone.utc)

    def fail(self) -> None:
        self.state = AgentState.FAILED
        self.ended_at = datetime.now(timezone.utc)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def add_tokens(self, prompt: int, completion: int) -> None:
        self.total_prompt_tokens += prompt
        self.total_completion_tokens += completion

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def elapsed_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.ended_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()

    def get(self, key: str, default: Any = None) -> Any:
        return self.store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value

    def fork(self) -> "AgentContext":
        """Create a child context that inherits session and messages."""
        child = AgentContext(
            session_id=self.session_id,
            messages=list(self.messages),
            max_iterations=self.max_iterations,
            metadata=dict(self.metadata),
        )
        return child
