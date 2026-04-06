"""Parallelization pattern — Chapter 3.

Runs multiple LLM calls or agent tasks concurrently using asyncio.gather,
then aggregates results (voting, summarization, or custom reduce).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from agentic.core.message import Message
from agentic.patterns.base import BasePattern, PatternResult

if TYPE_CHECKING:
    from agentic.core.context import AgentContext
    from agentic.llm.base import BaseLLMProvider


Task = Callable[[str, Any], Awaitable[str]]


class ParallelizationPattern(BasePattern):
    """Fan-out: execute multiple tasks in parallel, then aggregate results.

    Two built-in aggregation strategies:
    - ``"concat"``: join all outputs with a separator
    - ``"vote"``: ask the LLM to pick the best answer (majority vote)
    - ``"summarize"``: ask the LLM to synthesize all outputs
    - Custom: pass an ``aggregate_fn`` callable
    """

    def __init__(
        self,
        llm: "BaseLLMProvider | None" = None,
        tasks: list[Task] | None = None,
        aggregation: str = "concat",
        separator: str = "\n\n---\n\n",
        aggregate_fn: Callable[[list[str]], Awaitable[str]] | None = None,
        max_concurrency: int = 10,
    ) -> None:
        self.llm = llm
        self.tasks: list[Task] = tasks or []
        self.aggregation = aggregation
        self.separator = separator
        self.aggregate_fn = aggregate_fn
        self.semaphore = asyncio.Semaphore(max_concurrency)

    def add_task(self, task: Task) -> "ParallelizationPattern":
        self.tasks.append(task)
        return self

    async def _run_task(self, task: Task, input: str, context: Any) -> str:
        async with self.semaphore:
            return await task(input, context)

    async def _aggregate(self, outputs: list[str], input: str) -> str:
        if self.aggregate_fn:
            return await self.aggregate_fn(outputs)

        if self.aggregation == "concat":
            return self.separator.join(outputs)

        if self.aggregation == "vote" and self.llm:
            numbered = "\n".join(f"{i+1}. {o}" for i, o in enumerate(outputs))
            prompt = (
                f"Given the following answers to '{input}', pick the most accurate one "
                f"and explain why. Numbered answers:\n{numbered}\n\nBest answer number and reason:"
            )
            response = await self.llm.generate([Message.human(prompt)])
            return response.message.content

        if self.aggregation == "summarize" and self.llm:
            numbered = "\n".join(f"Response {i+1}: {o}" for i, o in enumerate(outputs))
            prompt = (
                f"Synthesize the following responses into a single comprehensive answer "
                f"to: '{input}'\n\n{numbered}\n\nSynthesized answer:"
            )
            response = await self.llm.generate([Message.human(prompt)])
            return response.message.content

        # Fallback: concatenate
        return self.separator.join(outputs)

    async def run(
        self,
        input: str,
        context: "AgentContext | None" = None,
        **kwargs: Any,
    ) -> PatternResult:
        if not self.tasks:
            return PatternResult(output="", success=False, error="No tasks configured.")

        # Run all tasks concurrently
        raw_results = await asyncio.gather(
            *[self._run_task(t, input, context) for t in self.tasks],
            return_exceptions=True,
        )

        outputs = []
        errors = []
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                errors.append(f"Task {i} failed: {r}")
                outputs.append(f"[Task {i} failed: {r}]")
            else:
                outputs.append(str(r))

        aggregated = await self._aggregate(outputs, input)

        return PatternResult(
            output=aggregated,
            steps=[f"Task {i}: {o[:100]}..." for i, o in enumerate(outputs)],
            metadata={
                "n_tasks": len(self.tasks),
                "n_errors": len(errors),
                "aggregation": self.aggregation,
                "errors": errors,
            },
            success=len(errors) < len(self.tasks),
        )


class MapReducePattern(BasePattern):
    """Map-reduce: apply a map prompt to each chunk, then reduce all results.

    Useful for processing long documents in parallel chunks.
    """

    def __init__(
        self,
        llm: "BaseLLMProvider",
        map_prompt: str = "Analyze the following text:\n{chunk}",
        reduce_prompt: str = "Combine these analyses into a unified summary:\n{results}",
        chunk_size: int = 1000,
        max_concurrency: int = 5,
    ) -> None:
        self.llm = llm
        self.map_prompt = map_prompt
        self.reduce_prompt = reduce_prompt
        self.chunk_size = chunk_size
        self.semaphore = asyncio.Semaphore(max_concurrency)

    def _chunk_text(self, text: str) -> list[str]:
        return [text[i: i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    async def _map_chunk(self, chunk: str) -> str:
        async with self.semaphore:
            prompt = self.map_prompt.replace("{chunk}", chunk)
            response = await self.llm.generate([Message.human(prompt)])
            return response.message.content

    async def run(
        self,
        input: str,
        context: "AgentContext | None" = None,
        **kwargs: Any,
    ) -> PatternResult:
        chunks = self._chunk_text(input)
        mapped = await asyncio.gather(*[self._map_chunk(c) for c in chunks])

        combined = "\n\n".join(mapped)
        reduce_prompt = self.reduce_prompt.replace("{results}", combined)
        final_response = await self.llm.generate([Message.human(reduce_prompt)])

        return PatternResult(
            output=final_response.message.content,
            steps=[f"Chunk {i}: {m[:100]}..." for i, m in enumerate(mapped)],
            metadata={"n_chunks": len(chunks)},
        )
