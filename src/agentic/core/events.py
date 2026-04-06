"""EventBus for cross-cutting observability and hook support."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Coroutine

from pydantic import BaseModel


# ── Event types ───────────────────────────────────────────────────────────────


class Event(BaseModel):
    """Base event emitted by the framework."""

    name: str
    payload: dict[str, Any] = {}


class AgentStartEvent(Event):
    name: str = "agent.start"


class AgentEndEvent(Event):
    name: str = "agent.end"


class AgentErrorEvent(Event):
    name: str = "agent.error"


class ToolCallEvent(Event):
    name: str = "tool.call"


class ToolResultEvent(Event):
    name: str = "tool.result"


class LLMCallEvent(Event):
    name: str = "llm.call"


class LLMResponseEvent(Event):
    name: str = "llm.response"


class PatternStartEvent(Event):
    name: str = "pattern.start"


class PatternEndEvent(Event):
    name: str = "pattern.end"


class GuardrailEvent(Event):
    name: str = "guardrail.triggered"


class MemoryEvent(Event):
    name: str = "memory.operation"


# ── Handler type aliases ──────────────────────────────────────────────────────

SyncHandler = Callable[[Event], None]
AsyncHandler = Callable[[Event], Coroutine[Any, Any, None]]
Handler = SyncHandler | AsyncHandler


# ── EventBus ─────────────────────────────────────────────────────────────────


class EventBus:
    """Simple publish/subscribe event bus supporting sync and async handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._global_handlers: list[Handler] = []

    def subscribe(self, event_name: str, handler: Handler) -> None:
        """Subscribe a handler to a specific event name."""
        self._handlers[event_name].append(handler)

    def subscribe_all(self, handler: Handler) -> None:
        """Subscribe a handler to ALL events."""
        self._global_handlers.append(handler)

    def unsubscribe(self, event_name: str, handler: Handler) -> None:
        self._handlers[event_name] = [
            h for h in self._handlers[event_name] if h is not handler
        ]

    async def emit(self, event: Event) -> None:
        """Emit an event, calling all subscribed handlers."""
        handlers = self._global_handlers + self._handlers.get(event.name, [])
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)

    def emit_sync(self, event: Event) -> None:
        """Synchronous emit — runs async handlers via asyncio.run if needed."""
        try:
            loop = asyncio.get_running_loop()
            # Schedule on the running loop
            loop.create_task(self.emit(event))
        except RuntimeError:
            asyncio.run(self.emit(event))


# Global default bus
_default_bus = EventBus()


def get_event_bus() -> EventBus:
    return _default_bus
