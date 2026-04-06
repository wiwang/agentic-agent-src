"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from pydantic import BaseModel

from agentic.core.message import Message, ToolCall


class LLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    message: Message
    usage: LLMUsage = LLMUsage()
    model: str = ""
    finish_reason: str = ""
    metadata: dict[str, Any] = {}


class ToolSchema(BaseModel):
    """JSON-schema descriptor for a tool, passed to the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]


class BaseLLMProvider(ABC):
    """Abstract interface for LLM providers.

    All methods are async to support non-blocking I/O.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._extra: dict[str, Any] = kwargs

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response for the given messages."""

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream response tokens."""

    async def generate_with_retry(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> LLMResponse:
        """Wrapper around generate() with exponential-backoff retry."""
        import asyncio

        from agentic.exceptions import LLMRateLimitError

        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return await self.generate(messages, tools=tools, **kwargs)
            except LLMRateLimitError as exc:
                last_exc = exc
                await asyncio.sleep(delay)
                delay *= 2
            except Exception:
                raise
        raise last_exc or RuntimeError("LLM generate failed after retries")

    def _messages_to_dicts(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert framework messages to provider-agnostic dicts (subclasses may override)."""
        result = []
        for m in messages:
            result.append({"role": m.role.value, "content": m.content})
        return result
