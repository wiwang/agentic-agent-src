"""Agent-to-Agent (A2A) communication subsystem."""

from agentic.a2a.agent_card import AgentCard, AgentCapability, AgentRegistry
from agentic.a2a.protocol import A2AClient, A2AServer, A2ATask, A2AMessage, TaskState

__all__ = [
    "AgentCard",
    "AgentCapability",
    "AgentRegistry",
    "A2AClient",
    "A2AServer",
    "A2ATask",
    "A2AMessage",
    "TaskState",
]
