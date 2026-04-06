"""Tests for agentic design patterns."""

from __future__ import annotations

import pytest

from tests.conftest import MockLLMProvider


# ── PromptChainingPattern ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_prompt_chaining_basic(mock_llm: MockLLMProvider) -> None:
    from agentic.patterns.prompt_chaining import PromptChainingPattern

    chain = PromptChainingPattern(llm=mock_llm)
    chain.add_step("step1", "Process: {input}")
    chain.add_step("step2", "Refine: {previous}")

    result = await chain.run("test input")
    assert result.success
    assert result.output == mock_llm.response_text
    assert result.metadata["n_steps"] == 2
    assert mock_llm.call_count == 2


@pytest.mark.asyncio
async def test_prompt_chaining_step_references(mock_llm: MockLLMProvider) -> None:
    from agentic.patterns.prompt_chaining import PromptChainingPattern

    chain = PromptChainingPattern(llm=mock_llm)
    chain.add_step("a", "Input was: {input}")
    chain.add_step("b", "Step 0 was: {step_0}, previous: {previous}")

    result = await chain.run("hello")
    assert result.success


# ── ReflectionPattern ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reflection_approves(mock_llm: MockLLMProvider) -> None:
    from agentic.patterns.reflection import ReflectionPattern

    # Make critic always approve
    mock_llm.response_text = "APPROVED: Looks great!"
    pattern = ReflectionPattern(producer_llm=mock_llm, critic_llm=mock_llm, max_iterations=3)
    result = await pattern.run("Write something")
    assert result.metadata["approved"] is True
    assert result.metadata["iterations"] == 1


@pytest.mark.asyncio
async def test_reflection_max_iterations(mock_llm: MockLLMProvider) -> None:
    from agentic.patterns.reflection import ReflectionPattern

    # Critic never approves
    mock_llm.response_text = "IMPROVE: Not good enough."
    pattern = ReflectionPattern(
        producer_llm=mock_llm, critic_llm=mock_llm, max_iterations=2
    )
    result = await pattern.run("Write something")
    assert result.metadata["iterations"] == 2
    assert result.metadata["approved"] is False


# ── ParallelizationPattern ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parallelization_concat() -> None:
    from agentic.patterns.parallelization import ParallelizationPattern

    async def task_a(inp: str, ctx: object) -> str:
        return f"A: {inp}"

    async def task_b(inp: str, ctx: object) -> str:
        return f"B: {inp}"

    pattern = ParallelizationPattern(tasks=[task_a, task_b], aggregation="concat")
    result = await pattern.run("hello")
    assert "A: hello" in result.output
    assert "B: hello" in result.output
    assert result.metadata["n_tasks"] == 2


@pytest.mark.asyncio
async def test_parallelization_handles_errors() -> None:
    from agentic.patterns.parallelization import ParallelizationPattern

    async def good_task(inp: str, ctx: object) -> str:
        return "success"

    async def bad_task(inp: str, ctx: object) -> str:
        raise ValueError("boom")

    pattern = ParallelizationPattern(tasks=[good_task, bad_task], aggregation="concat")
    result = await pattern.run("test")
    assert result.metadata["n_errors"] == 1
    # success is False only when ALL tasks fail; partial failure still returns output
    assert "success" in result.output


# ── LLM/Rule-Based Router ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rule_based_router() -> None:
    from agentic.patterns.routing import RuleBasedRouter

    calls: list[str] = []

    async def math_handler(inp: str, ctx: object) -> str:
        calls.append("math")
        return "math answer"

    async def code_handler(inp: str, ctx: object) -> str:
        calls.append("code")
        return "code answer"

    router = RuleBasedRouter()
    router.add_route("math", "Math questions", math_handler, keywords=[r"\d+", "calculate"])
    router.add_route("code", "Code questions", code_handler, keywords=["python", "function"])

    result = await router.run("calculate 2 + 2")
    assert result.metadata["route"] == "math"
    assert "math" in calls

    result = await router.run("write a python function")
    assert result.metadata["route"] == "code"


# ── SequentialPattern ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sequential_pattern(mock_llm: MockLLMProvider) -> None:
    from agentic.agents import ToolAgent
    from agentic.patterns.multi_agent import SequentialPattern

    agent1 = ToolAgent(llm=mock_llm, agent_id="agent1")
    agent2 = ToolAgent(llm=mock_llm, agent_id="agent2")

    sequential = SequentialPattern(agents=[agent1, agent2])
    result = await sequential.run("test input")
    assert result.output == mock_llm.response_text
    assert result.metadata["n_agents"] == 2
    # Both agents should have been called
    assert mock_llm.call_count == 2
