"""Agent Cards for agent discovery in A2A (Agent-to-Agent) communication.

An Agent Card describes an agent's capabilities, endpoints, and metadata,
enabling other agents to discover and interact with it dynamically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AgentCapability(BaseModel):
    """A specific capability an agent exposes."""

    name: str
    description: str
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}


class AgentCard(BaseModel):
    """Standard descriptor for an agent's identity and capabilities.

    Based on the A2A Protocol spec (Google, 2025).
    """

    name: str
    description: str
    version: str = "0.1.0"
    url: str = ""  # HTTP endpoint for A2A communication
    capabilities: list[AgentCapability] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Authentication
    auth_type: str = "none"  # "none", "bearer", "api_key"
    auth_header: str = ""

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> "AgentCard":
        return cls.model_validate_json(data)

    def save(self, path: str) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str) -> "AgentCard":
        return cls.from_json(Path(path).read_text())

    def has_capability(self, name: str) -> bool:
        return any(c.name == name for c in self.capabilities)

    def get_capability(self, name: str) -> AgentCapability | None:
        for c in self.capabilities:
            if c.name == name:
                return c
        return None


class AgentRegistry:
    """Local registry for discovered agent cards."""

    def __init__(self) -> None:
        self._cards: dict[str, AgentCard] = {}

    def register(self, card: AgentCard) -> None:
        self._cards[card.name] = card

    def find_by_capability(self, capability: str) -> list[AgentCard]:
        return [c for c in self._cards.values() if c.has_capability(capability)]

    def find_by_skill(self, skill: str) -> list[AgentCard]:
        return [c for c in self._cards.values() if skill in c.skills]

    def get(self, name: str) -> AgentCard | None:
        return self._cards.get(name)

    def all_cards(self) -> list[AgentCard]:
        return list(self._cards.values())
