"""Multi-Agent patterns — Chapter 7.

Implements multiple topologies for coordinating groups of agents:
- SupervisorPattern: one orchestrator delegates to specialist agents
- HierarchicalPattern: tree of supervisors and sub-agents
- NetworkPattern: peer agents communicate in a mesh
- SequentialPattern: agents process in a fixed pipeline
- ParallelPattern: agents run concurrently, results merged
"""

from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

from agentic.core.message import Message
from agentic.patterns.base import BasePattern, PatternResult

if TYPE_CHECKING:
    from agentic.core.context import AgentContext
    from agentic.core.agent import BaseAgent
    from agentic.llm.base import BaseLLMProvider


# ── Supervisor ────────────────────────────────────────────────────────────────

SUPERVISOR_PROMPT = """You are a supervisor agent. Delegate subtasks to specialist agents and aggregate results.

Specialist agents:
{agents}

User request: {request}

Decide which agent(s) to use and what to ask them.
Respond with a JSON list of delegations:
[
  {{"agent": "<agent_name>", "task": "<specific task for that agent>"}},
  ...
]

JSON only:"""

AGGREGATOR_PROMPT = """Synthesize the following agent responses into a final answer.

Original request: {request}

Agent responses:
{responses}

Final answer:"""


class AgentSpec(BaseModel):
    name: str
    description: str


class SupervisorPattern(BasePattern):
    """One LLM supervisor delegates tasks to named specialist agents."""

    def __init__(
        self,
        supervisor_llm: "BaseLLMProvider",
        agents: dict[str, "BaseAgent"] | None = None,
    ) -> None:
        self.supervisor_llm = supervisor_llm
        self.agents: dict[str, "BaseAgent"] = agents or {}

    def register_agent(self, name: str, agent: "BaseAgent") -> "SupervisorPattern":
        self.agents[name] = agent
        return self

    async def _delegate(self, request: str) -> list[dict[str, str]]:
        import json, re
        agent_list = "\n".join(f"- {n}: {a.agent_id}" for n, a in self.agents.items())
        prompt = SUPERVISOR_PROMPT.format(agents=agent_list, request=request)
        response = await self.supervisor_llm.generate([Message.human(prompt)])
        text = response.message.content.strip()
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        try:
            return json.loads(text)
        except Exception:
            # Fallback: use all agents
            return [{"agent": n, "task": request} for n in self.agents]

    async def run(
        self,
        input: str,
        context: "AgentContext | None" = None,
        **kwargs: Any,
    ) -> PatternResult:
        delegations = await self._delegate(input)
        tasks = []
        for d in delegations:
            agent = self.agents.get(d.get("agent", ""))
            if agent:
                tasks.append((d["agent"], agent.run(d.get("task", input))))

        results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
        responses = []
        for (name, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                responses.append(f"{name}: [Error: {result}]")
            else:
                responses.append(f"{name}: {result.content}")  # type: ignore[union-attr]

        agg_prompt = AGGREGATOR_PROMPT.format(
            request=input, responses="\n\n".join(responses)
        )
        final = await self.supervisor_llm.generate([Message.human(agg_prompt)])

        return PatternResult(
            output=final.message.content,
            steps=[f"Delegated to: {d.get('agent')}" for d in delegations] + responses,
            metadata={"delegations": delegations},
        )


# ── Sequential ────────────────────────────────────────────────────────────────

class SequentialPattern(BasePattern):
    """Agents run in a fixed pipeline: output of agent N → input of agent N+1."""

    def __init__(self, agents: list["BaseAgent"] | None = None) -> None:
        self.agents: list["BaseAgent"] = agents or []

    def add_agent(self, agent: "BaseAgent") -> "SequentialPattern":
        self.agents.append(agent)
        return self

    async def run(
        self,
        input: str,
        context: "AgentContext | None" = None,
        **kwargs: Any,
    ) -> PatternResult:
        current = input
        steps = []
        for agent in self.agents:
            child_ctx = context.fork() if context else None
            result = await agent.run(current, context=child_ctx)
            steps.append(f"{agent.agent_id}: {result.content[:150]}...")
            current = result.content

        return PatternResult(output=current, steps=steps, metadata={"n_agents": len(self.agents)})


# ── Parallel ──────────────────────────────────────────────────────────────────

class ParallelAgentPattern(BasePattern):
    """All agents run concurrently on the same input; results are merged."""

    def __init__(
        self,
        agents: list["BaseAgent"] | None = None,
        aggregator_llm: "BaseLLMProvider | None" = None,
    ) -> None:
        self.agents: list["BaseAgent"] = agents or []
        self.aggregator_llm = aggregator_llm

    def add_agent(self, agent: "BaseAgent") -> "ParallelAgentPattern":
        self.agents.append(agent)
        return self

    async def run(
        self,
        input: str,
        context: "AgentContext | None" = None,
        **kwargs: Any,
    ) -> PatternResult:
        tasks = [a.run(input, context=context.fork() if context else None) for a in self.agents]
        raw = await asyncio.gather(*tasks, return_exceptions=True)

        responses = []
        for agent, result in zip(self.agents, raw):
            if isinstance(result, Exception):
                responses.append(f"{agent.agent_id}: [Error: {result}]")
            else:
                responses.append(f"{agent.agent_id}: {result.content}")  # type: ignore

        if self.aggregator_llm:
            prompt = AGGREGATOR_PROMPT.format(request=input, responses="\n\n".join(responses))
            agg = await self.aggregator_llm.generate([Message.human(prompt)])
            output = agg.message.content
        else:
            output = "\n\n---\n\n".join(responses)

        return PatternResult(
            output=output,
            steps=responses,
            metadata={"n_agents": len(self.agents)},
        )


# ── Network (mesh) ────────────────────────────────────────────────────────────

class NetworkPattern(BasePattern):
    """Peer agents communicate in a mesh: each agent sees all others' outputs.

    Runs for a fixed number of rounds; agents share context across rounds.
    """

    def __init__(
        self,
        agents: list["BaseAgent"] | None = None,
        rounds: int = 2,
        aggregator_llm: "BaseLLMProvider | None" = None,
    ) -> None:
        self.agents: list["BaseAgent"] = agents or []
        self.rounds = rounds
        self.aggregator_llm = aggregator_llm

    async def run(
        self,
        input: str,
        context: "AgentContext | None" = None,
        **kwargs: Any,
    ) -> PatternResult:
        shared_context = input
        steps = []

        for round_n in range(self.rounds):
            tasks = [a.run(shared_context, context=context.fork() if context else None) for a in self.agents]
            raw = await asyncio.gather(*tasks, return_exceptions=True)

            round_outputs = []
            for agent, result in zip(self.agents, raw):
                if isinstance(result, Exception):
                    round_outputs.append(f"{agent.agent_id}: [Error: {result}]")
                else:
                    round_outputs.append(f"{agent.agent_id}: {result.content}")  # type: ignore

            # Build shared context for next round
            shared_context = (
                f"Original task: {input}\n\nRound {round_n+1} results:\n"
                + "\n\n".join(round_outputs)
            )
            steps.extend(round_outputs)

        if self.aggregator_llm:
            prompt = f"Synthesize the multi-round agent discussion into a final answer for: '{input}'\n\n{shared_context}"
            final = await self.aggregator_llm.generate([Message.human(prompt)])
            output = final.message.content
        else:
            output = shared_context

        return PatternResult(
            output=output, steps=steps, metadata={"rounds": self.rounds, "n_agents": len(self.agents)}
        )
