"""Concrete agent implementations built on BaseAgent."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from agentic.core.agent import BaseAgent
from agentic.core.context import AgentContext
from agentic.core.message import Message
from agentic.llm.base import ToolSchema

if TYPE_CHECKING:
    from agentic.llm.base import BaseLLMProvider
    from agentic.memory.base import BaseMemory
    from agentic.core.events import EventBus
    from agentic.reasoning.base import BaseReasoner


class ToolAgent(BaseAgent):
    """A general-purpose agent that can use tools via function calling.

    This is the standard "ReAct-style" agent — it generates responses
    and invokes tools when needed, looping until it produces a final answer.

    Example::

        llm = OpenAIProvider(model="gpt-4o")
        agent = ToolAgent(llm=llm, system_prompt="You are a helpful assistant.")
        agent.add_tool(CalculatorTool())
        result = await agent.run("What is 2^32?")
    """

    def __init__(
        self,
        llm: "BaseLLMProvider",
        agent_id: str = "ToolAgent",
        memory: "BaseMemory | None" = None,
        event_bus: "EventBus | None" = None,
        max_iterations: int = 20,
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            llm=llm,
            memory=memory,
            event_bus=event_bus,
            max_iterations=max_iterations,
            system_prompt=system_prompt,
        )

    async def step(self, messages: list[Message], context: AgentContext) -> Message:
        """One LLM turn: generate a response, optionally requesting tool calls."""
        if self.llm is None:
            raise ValueError("ToolAgent requires an LLM provider.")

        # Build tool schemas
        tool_schemas: list[ToolSchema] = [
            t.to_schema() for t in self._tools.values()
        ] if self._tools else []

        response = await self.llm.generate(
            messages,
            tools=tool_schemas if tool_schemas else None,
        )
        # Track token usage
        context.add_tokens(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
        return response.message


class ReasoningAgent(BaseAgent):
    """An agent that applies a pluggable reasoning strategy before each LLM call.

    Supports CoT, ToT, and ReAct reasoning strategies.

    Example::

        llm = OpenAIProvider(model="gpt-4o")
        reasoner = ChainOfThoughtReasoner(llm=llm)
        agent = ReasoningAgent(llm=llm, reasoner=reasoner)
        result = await agent.run("Solve this logic puzzle: ...")
    """

    def __init__(
        self,
        llm: "BaseLLMProvider",
        reasoner: "BaseReasoner",
        agent_id: str = "ReasoningAgent",
        memory: "BaseMemory | None" = None,
        event_bus: "EventBus | None" = None,
        max_iterations: int = 5,
        system_prompt: str = "You are a careful reasoning assistant.",
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            llm=llm,
            memory=memory,
            event_bus=event_bus,
            max_iterations=max_iterations,
            system_prompt=system_prompt,
        )
        self.reasoner = reasoner

    async def step(self, messages: list[Message], context: AgentContext) -> Message:
        """Apply the reasoning strategy to the latest user message."""
        # Find the last human message
        last_human = next(
            (m for m in reversed(messages) if m.role.value == "human"), None
        )
        question = last_human.content if last_human else "Continue."

        result = await self.reasoner.reason(question, context=context)
        return Message.ai(content=result.answer)


class ConversationalAgent(BaseAgent):
    """A simple conversational agent that maintains chat history.

    No tool calls — focused on multi-turn dialogue.
    """

    def __init__(
        self,
        llm: "BaseLLMProvider",
        agent_id: str = "ConversationalAgent",
        memory: "BaseMemory | None" = None,
        system_prompt: str = "You are a friendly conversational assistant.",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            llm=llm,
            memory=memory,
            system_prompt=system_prompt,
            **kwargs,
        )

    async def step(self, messages: list[Message], context: AgentContext) -> Message:
        if self.llm is None:
            raise ValueError("ConversationalAgent requires an LLM provider.")
        response = await self.llm.generate(messages)
        context.add_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.message
