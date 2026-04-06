"""Message types used throughout the agentic framework."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    HUMAN = "human"
    AI = "ai"
    TOOL = "tool"


class ToolCall(BaseModel):
    """Represents a tool invocation requested by the AI."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result returned after executing a tool call."""

    tool_call_id: str
    name: str
    content: str
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.error is not None


class Message(BaseModel):
    """A single message in a conversation."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Role
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def system(cls, content: str, **metadata: Any) -> "Message":
        return cls(role=Role.SYSTEM, content=content, metadata=metadata)

    @classmethod
    def human(cls, content: str, **metadata: Any) -> "Message":
        return cls(role=Role.HUMAN, content=content, metadata=metadata)

    @classmethod
    def ai(
        cls,
        content: str,
        tool_calls: list[ToolCall] | None = None,
        **metadata: Any,
    ) -> "Message":
        return cls(
            role=Role.AI,
            content=content,
            tool_calls=tool_calls or [],
            metadata=metadata,
        )

    @classmethod
    def tool(cls, result: ToolResult, **metadata: Any) -> "Message":
        return cls(
            role=Role.TOOL,
            content=result.content,
            tool_results=[result],
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (useful for LLM API calls)."""
        return self.model_dump(mode="json")
