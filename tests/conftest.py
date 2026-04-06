"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic.core.context import AgentContext
from agentic.core.message import Message, Role
from agentic.llm.base import BaseLLMProvider, LLMResponse, LLMUsage, ToolSchema


class MockLLMProvider(BaseLLMProvider):
    """A mock LLM provider for testing — returns configurable responses."""

    def __init__(self, response_text: str = "Mock response", model: str = "mock") -> None:
        super().__init__(model=model, temperature=0.0, max_tokens=100)
        self.response_text = response_text
        self.call_count = 0
        self.last_messages: list[Message] = []

    async def generate(
        self, messages: list[Message], tools: list[ToolSchema] | None = None, **kwargs
    ) -> LLMResponse:
        self.call_count += 1
        self.last_messages = messages
        return LLMResponse(
            message=Message.ai(content=self.response_text),
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model=self.model,
            finish_reason="stop",
        )

    async def stream(self, messages: list[Message], tools=None, **kwargs):
        for token in self.response_text.split():
            yield token + " "


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider(response_text="This is a test response.")


@pytest.fixture
def agent_context() -> AgentContext:
    return AgentContext(agent_id="test_agent", max_iterations=5)


@pytest.fixture
def sample_messages() -> list[Message]:
    return [
        Message.system("You are a helpful assistant."),
        Message.human("Hello, how are you?"),
    ]
