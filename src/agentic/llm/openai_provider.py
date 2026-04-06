"""OpenAI LLM provider implementation."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from agentic.core.message import Message, Role, ToolCall
from agentic.exceptions import LLMAuthError, LLMError, LLMRateLimitError
from agentic.llm.base import BaseLLMProvider, LLMResponse, LLMUsage, ToolSchema


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI chat completion models (GPT-4o, GPT-4, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("Install 'openai' to use OpenAIProvider.") from e

        self._client = AsyncOpenAI(api_key=api_key)

    def _to_openai_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        result = []
        for m in messages:
            if m.role == Role.SYSTEM:
                result.append({"role": "system", "content": m.content})
            elif m.role == Role.HUMAN:
                result.append({"role": "user", "content": m.content})
            elif m.role == Role.AI:
                entry: dict[str, Any] = {"role": "assistant", "content": m.content}
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                result.append(entry)
            elif m.role == Role.TOOL:
                for tr in m.tool_results:
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.tool_call_id,
                            "content": tr.content if not tr.error else tr.error,
                        }
                    )
        return result

    def _to_openai_tools(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        from openai import APIConnectionError, AuthenticationError, RateLimitError

        openai_messages = self._to_openai_messages(messages)
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs,
        }
        if tools:
            call_kwargs["tools"] = self._to_openai_tools(tools)
            call_kwargs["tool_choice"] = "auto"

        try:
            resp = await self._client.chat.completions.create(**call_kwargs)
        except AuthenticationError as e:
            raise LLMAuthError(str(e)) from e
        except RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except APIConnectionError as e:
            raise LLMError(str(e)) from e

        choice = resp.choices[0]
        raw = choice.message

        # Parse tool calls
        tool_calls: list[ToolCall] = []
        if raw.tool_calls:
            for tc in raw.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )

        ai_message = Message.ai(
            content=raw.content or "",
            tool_calls=tool_calls,
        )

        usage = LLMUsage(
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
            total_tokens=resp.usage.total_tokens if resp.usage else 0,
        )

        return LLMResponse(
            message=ai_message,
            usage=usage,
            model=resp.model,
            finish_reason=choice.finish_reason or "",
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        openai_messages = self._to_openai_messages(messages)
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            **kwargs,
        }

        async with self._client.chat.completions.stream(**call_kwargs) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
