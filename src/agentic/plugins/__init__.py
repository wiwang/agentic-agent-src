"""Plugin subsystem for the agentic framework."""

from agentic.plugins.base import AgentPlugin, PluginMetadata
from agentic.plugins.manager import PluginManager, get_plugin_manager

__all__ = [
    "AgentPlugin",
    "PluginMetadata",
    "PluginManager",
    "get_plugin_manager",
]
