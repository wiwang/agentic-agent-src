"""BaseAgent — abstract foundation for all agents in the framework."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from agentic.core.context import AgentContext
from agentic.core.events import (
    AgentEndEvent,
    AgentErrorEvent,
    AgentStartEvent,
    EventBus,
    get_event_bus,
)
from agentic.core.message import Message, Role
from agentic.exceptions import AgentError

if TYPE_CHECKING:
    from agentic.guardrails.base import BaseGuardrail
    from agentic.llm.base import BaseLLMProvider
    from agentic.memory.base import BaseMemory
    from agentic.tools.base import BaseTool


class AgentResponse(BaseModel):
    """The final structured response from an agent run."""

    content: str
    messages: list[Message] = []
    metadata: dict[str, Any] = {}
    run_id: str = ""
    total_tokens: int = 0
    elapsed_seconds: float | None = None


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Subclasses must implement:
        - ``step(messages) -> Message``   (single LLM turn)

    ``run()`` handles the overall loop, guardrails, memory, and events.
    """

    def __init__(
        self,
        agent_id: str = "",
        llm: "BaseLLMProvider | None" = None,
        memory: "BaseMemory | None" = None,
        event_bus: EventBus | None = None,
        max_iterations: int = 20,
        system_prompt: str = "",
    ) -> None:
        self.agent_id = agent_id or self.__class__.__name__
        self.llm = llm
        self.memory = memory
        self.event_bus = event_bus or get_event_bus()
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt

        self._tools: dict[str, "BaseTool"] = {}
        self._guardrails: list["BaseGuardrail"] = []

    # ── Tool management ────────────────────────────────────────────────────

    def add_tool(self, tool: "BaseTool") -> "BaseAgent":
        self._tools[tool.name] = tool
        return self

    def remove_tool(self, name: str) -> None:
        self._tools.pop(name, None)

    @property
    def tools(self) -> dict[str, "BaseTool"]:
        return dict(self._tools)

    # ── Guardrail management ───────────────────────────────────────────────

    def add_guardrail(self, guardrail: "BaseGuardrail") -> "BaseAgent":
        self._guardrails.append(guardrail)
        return self

    # ── Core interface ─────────────────────────────────────────────────────

    @abstractmethod
    async def step(self, messages: list[Message], context: AgentContext) -> Message:
        """Execute one reasoning step and return the AI message."""

    async def run(
        self,
        input: str,
        context: AgentContext | None = None,
        **kwargs: Any,
    ) -> AgentResponse:
        """Run the agent to completion with full lifecycle management."""
        ctx = context or AgentContext(
            agent_id=self.agent_id,
            max_iterations=self.max_iterations,
        )
        ctx.agent_id = self.agent_id
        ctx.start()

        await self.event_bus.emit(
            AgentStartEvent(payload={"agent_id": self.agent_id, "run_id": ctx.run_id})
        )

        try:
            # Apply input guardrails
            processed_input = await self._run_input_guardrails(input)

            # Inject system prompt
            if self.system_prompt and not any(
                m.role == Role.SYSTEM for m in ctx.messages
            ):
                ctx.add_message(Message.system(self.system_prompt))

            # Load relevant memories
            if self.memory:
                memories = await self.memory.retrieve(processed_input)
                if memories:
                    ctx.add_message(
                        Message.system(
                            "Relevant memories:\n" + "\n".join(memories)
                        )
                    )

            ctx.add_message(Message.human(processed_input))

            # Main reasoning loop
            response_message = await self._run_loop(ctx, **kwargs)

            # Apply output guardrails
            final_content = await self._run_output_guardrails(
                response_message.content
            )

            # Store in memory
            if self.memory:
                await self.memory.store(
                    input=processed_input, output=final_content, context=ctx
                )

            ctx.complete()

            await self.event_bus.emit(
                AgentEndEvent(
                    payload={"agent_id": self.agent_id, "run_id": ctx.run_id}
                )
            )

            return AgentResponse(
                content=final_content,
                messages=ctx.messages,
                run_id=ctx.run_id,
                total_tokens=ctx.total_tokens,
                elapsed_seconds=ctx.elapsed_seconds,
            )

        except Exception as exc:
            ctx.fail()
            await self.event_bus.emit(
                AgentErrorEvent(
                    payload={
                        "agent_id": self.agent_id,
                        "run_id": ctx.run_id,
                        "error": str(exc),
                    }
                )
            )
            raise AgentError(f"Agent '{self.agent_id}' failed: {exc}") from exc

    async def _run_loop(self, ctx: AgentContext, **kwargs: Any) -> Message:
        """Inner reasoning loop; subclasses can override for custom behaviour."""
        while ctx.iteration < ctx.max_iterations:
            ctx.iteration += 1
            response = await self.step(ctx.messages, ctx)
            ctx.add_message(response)

            # Execute tool calls if present
            if response.tool_calls:
                tool_messages = await self._execute_tool_calls(response, ctx)
                for tm in tool_messages:
                    ctx.add_message(tm)
                continue  # Let the LLM see the tool results

            # No tool calls → we have a final answer
            return response

        raise AgentError(
            f"Agent '{self.agent_id}' reached max iterations ({ctx.max_iterations})."
        )

    async def _execute_tool_calls(
        self, ai_message: Message, ctx: AgentContext
    ) -> list[Message]:
        from agentic.core.events import ToolCallEvent, ToolResultEvent
        from agentic.core.message import Message, ToolResult
        from agentic.exceptions import ToolNotFoundError

        tasks = []
        for tc in ai_message.tool_calls:
            tasks.append(self._call_tool(tc, ctx))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        messages = []
        for tc, result in zip(ai_message.tool_calls, results):
            if isinstance(result, Exception):
                tool_result = ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content="",
                    error=str(result),
                )
            else:
                tool_result = result  # type: ignore[assignment]

            await self.event_bus.emit(
                ToolResultEvent(
                    payload={
                        "tool": tc.name,
                        "run_id": ctx.run_id,
                        "error": tool_result.error,
                    }
                )
            )
            messages.append(Message.tool(tool_result))

        return messages

    async def _call_tool(self, tool_call: Any, ctx: AgentContext) -> Any:
        from agentic.core.events import ToolCallEvent
        from agentic.exceptions import ToolNotFoundError

        tool = self._tools.get(tool_call.name)
        if tool is None:
            raise ToolNotFoundError(tool_call.name)

        await self.event_bus.emit(
            ToolCallEvent(
                payload={"tool": tool_call.name, "run_id": ctx.run_id}
            )
        )
        return await tool.execute(tool_call.arguments, context=ctx)

    async def _run_input_guardrails(self, text: str) -> str:
        for guard in self._guardrails:
            if hasattr(guard, "check_input"):
                text = await guard.check_input(text)
        return text

    async def _run_output_guardrails(self, text: str) -> str:
        for guard in self._guardrails:
            if hasattr(guard, "check_output"):
                text = await guard.check_output(text)
        return text
