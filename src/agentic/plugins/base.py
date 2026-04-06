"""Plugin base interface for the agentic framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from agentic.tools.base import BaseTool
    from agentic.memory.base import BaseMemory
    from agentic.guardrails.base import BaseGuardrail


class PluginMetadata(BaseModel):
    """Metadata describing a plugin."""

    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    tags: list[str] = []


class AgentPlugin(ABC):
    """Abstract interface for agentic framework plugins.

    A plugin can contribute:
    - Tools (``get_tools``)
    - Memory implementations (``get_memory``)
    - Guardrails (``get_guardrails``)
    - Arbitrary configuration (``configure``)

    Plugin discovery is done via setuptools entry_points or directory scanning.
    """

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""

    def get_tools(self) -> list["BaseTool"]:
        """Return tools contributed by this plugin."""
        return []

    def get_memory(self) -> "BaseMemory | None":
        """Return a memory implementation (optional)."""
        return None

    def get_guardrails(self) -> list["BaseGuardrail"]:
        """Return guardrails contributed by this plugin."""
        return []

    def configure(self, config: dict[str, Any]) -> None:
        """Apply configuration to the plugin (called during load)."""

    def on_load(self) -> None:
        """Called after the plugin is successfully loaded."""

    def on_unload(self) -> None:
        """Called before the plugin is unloaded."""
