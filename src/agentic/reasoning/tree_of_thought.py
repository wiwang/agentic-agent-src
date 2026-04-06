"""Tree-of-Thought (ToT) reasoning strategy.

Implements the ToT framework (Yao et al. 2023) where the LLM explores multiple
reasoning branches and selects the most promising path using BFS or DFS.
"""

from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from agentic.core.message import Message
from agentic.reasoning.base import BaseReasoner, ReasoningResult

if TYPE_CHECKING:
    from agentic.core.context import AgentContext
    from agentic.llm.base import BaseLLMProvider


class ThoughtNode(BaseModel):
    """A single node in the thought tree."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    thought: str
    score: float = 0.0
    depth: int = 0
    parent_thought: str = ""
    children: list["ThoughtNode"] = []


GENERATE_THOUGHTS_PROMPT = """You are solving a problem step by step. Generate {n} diverse next thoughts/steps.
Each thought should be on its own line prefixed with "Thought: ".

Problem: {problem}
Current reasoning so far:
{current_thoughts}

Generate {n} possible next thoughts:"""

EVALUATE_THOUGHT_PROMPT = """Evaluate how promising this reasoning path is for solving the problem.
Rate from 0.0 (dead end) to 1.0 (very promising).
Respond with ONLY a number between 0.0 and 1.0.

Problem: {problem}
Reasoning path: {thought_path}

Score:"""

FINAL_ANSWER_PROMPT = """Based on the following reasoning path, provide a final answer.

Problem: {problem}
Reasoning: {best_path}

Final answer:"""


class TreeOfThoughtReasoner(BaseReasoner):
    """Tree-of-Thought reasoning using BFS or DFS search over thought branches.

    Args:
        llm: LLM provider.
        n_thoughts: Number of thought branches to generate at each node.
        max_depth: Maximum depth of the thought tree.
        search: "bfs" (breadth-first) or "dfs" (depth-first).
        beam_width: For BFS, number of top nodes to expand at each level.
    """

    def __init__(
        self,
        llm: "BaseLLMProvider",
        n_thoughts: int = 3,
        max_depth: int = 3,
        search: str = "bfs",
        beam_width: int = 2,
    ) -> None:
        super().__init__(llm)
        self.n_thoughts = n_thoughts
        self.max_depth = max_depth
        self.search = search
        self.beam_width = beam_width

    async def _generate_thoughts(self, problem: str, path: list[str]) -> list[str]:
        current = "\n".join(path) if path else "(start)"
        prompt = GENERATE_THOUGHTS_PROMPT.format(
            n=self.n_thoughts, problem=problem, current_thoughts=current
        )
        response = await self.llm.generate([Message.human(prompt)])
        thoughts = []
        for line in response.message.content.split("\n"):
            line = line.strip()
            if line.lower().startswith("thought:"):
                thoughts.append(line[len("thought:"):].strip())
        # Fallback if parsing fails
        if not thoughts:
            thoughts = [t.strip() for t in response.message.content.split("\n") if t.strip()]
        return thoughts[: self.n_thoughts]

    async def _score_thought(self, problem: str, path: list[str]) -> float:
        prompt = EVALUATE_THOUGHT_PROMPT.format(
            problem=problem, thought_path="\n".join(path)
        )
        response = await self.llm.generate([Message.human(prompt)])
        try:
            score = float(response.message.content.strip().split()[0])
            return max(0.0, min(1.0, score))
        except (ValueError, IndexError):
            return 0.5

    async def _bfs(self, problem: str) -> tuple[list[str], float]:
        """Breadth-first search over thought tree."""
        # Each state: (path, cumulative_score)
        frontier: list[tuple[list[str], float]] = [([], 0.0)]
        best_path: list[str] = []
        best_score = -1.0

        for depth in range(self.max_depth):
            next_frontier: list[tuple[list[str], float]] = []
            tasks = []
            for path, _ in frontier:
                tasks.append(self._generate_thoughts(problem, path))
            all_thoughts = await asyncio.gather(*tasks)

            score_tasks = []
            candidate_paths = []
            for (path, _), thoughts in zip(frontier, all_thoughts):
                for thought in thoughts:
                    new_path = path + [thought]
                    candidate_paths.append(new_path)
                    score_tasks.append(self._score_thought(problem, new_path))

            scores = await asyncio.gather(*score_tasks)

            for new_path, score in zip(candidate_paths, scores):
                next_frontier.append((new_path, score))
                if score > best_score:
                    best_score = score
                    best_path = new_path

            # Keep top beam_width paths
            next_frontier.sort(key=lambda x: x[1], reverse=True)
            frontier = next_frontier[: self.beam_width]

        return best_path, best_score

    async def _dfs(
        self, problem: str, path: list[str] = [], depth: int = 0
    ) -> tuple[list[str], float]:
        """Depth-first search (greedy) over thought tree."""
        if depth >= self.max_depth:
            score = await self._score_thought(problem, path)
            return path, score

        thoughts = await self._generate_thoughts(problem, path)
        scores = await asyncio.gather(
            *[self._score_thought(problem, path + [t]) for t in thoughts]
        )

        # Pick the best thought and recurse
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return await self._dfs(
            problem, path + [thoughts[best_idx]], depth + 1
        )

    async def reason(
        self,
        question: str,
        context: "AgentContext | None" = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        if self.search == "bfs":
            best_path, best_score = await self._bfs(question)
        else:
            best_path, best_score = await self._dfs(question)

        # Generate final answer from best path
        final_prompt = FINAL_ANSWER_PROMPT.format(
            problem=question, best_path="\n".join(best_path)
        )
        final_response = await self.llm.generate([Message.human(final_prompt)])

        return ReasoningResult(
            answer=final_response.message.content,
            reasoning_trace=best_path,
            confidence=best_score,
            metadata={
                "search": self.search,
                "depth": len(best_path),
                "n_thoughts": self.n_thoughts,
            },
        )
