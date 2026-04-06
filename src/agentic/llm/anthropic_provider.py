"""Anthropic (Claude) LLM provider implementation."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from agentic.core.message import Message, Role, ToolCall
from agentic.exceptions import LLMAuthError, LLMError, LLMRateLimitError
from agentic.llm.base import BaseLLMProvider, LLMResponse, LLMUsage, ToolSchema


class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic Claude models."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ImportError("Install 'anthropic' to use AnthropicProvider.") from e

        self._client = AsyncAnthropic(api_key=api_key)

    def _split_messages(
        self, messages: list[Message]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Return (system_text, anthropic_messages)."""
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []

        for m in messages:
            if m.role == Role.SYSTEM:
                system_parts.append(m.content)
            elif m.role == Role.HUMAN:
                anthropic_messages.append({"role": "user", "content": m.content})
            elif m.role == Role.AI:
                content: list[Any] = []
                if m.content:
                    content.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                anthropic_messages.append(
                    {"role": "assistant", "content": content or m.content}
                )
            elif m.role == Role.TOOL:
                tool_results = []
                for tr in m.tool_results:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tr.tool_call_id,
                            "content": tr.error or tr.content,
                            "is_error": tr.error is not None,
                        }
                    )
                anthropic_messages.append({"role": "user", "content": tool_results})

        return "\n".join(system_parts), anthropic_messages

    def _to_anthropic_tools(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        from anthropic import APIConnectionError, AuthenticationError, RateLimitError

        system_text, anthropic_messages = self._split_messages(messages)

        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens,
            **kwargs,
        }
        if system_text:
            call_kwargs["system"] = system_text
        if tools:
            call_kwargs["tools"] = self._to_anthropic_tools(tools)

        try:
            resp = await self._client.messages.create(**call_kwargs)
        except AuthenticationError as e:
            raise LLMAuthError(str(e)) from e
        except RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except APIConnectionError as e:
            raise LLMError(str(e)) from e

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        ai_message = Message.ai(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
        )

        usage = LLMUsage(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            total_tokens=resp.usage.input_tokens + resp.usage.output_tokens,
        )

        return LLMResponse(
            message=ai_message,
            usage=usage,
            model=resp.model,
            finish_reason=resp.stop_reason or "",
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        system_text, anthropic_messages = self._split_messages(messages)

        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens,
            **kwargs,
        }
        if system_text:
            call_kwargs["system"] = system_text

        async with self._client.messages.stream(**call_kwargs) as stream:
            async for text in stream.text_stream:
                yield text
