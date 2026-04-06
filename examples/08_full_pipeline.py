"""Example 08: Full pipeline combining multiple patterns and features.

Demonstrates the complete agentic framework:
- ToolAgent with guardrails and memory
- ReflectionPattern for quality assurance
- Evaluation with LLMJudgeEvaluator
- Metrics collection
- Plugin system
- EventBus observability
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agentic import OpenAIProvider, ToolAgent
from agentic.core.events import get_event_bus, Event
from agentic.evaluation.evaluator import LLMJudgeEvaluator
from agentic.evaluation.metrics import get_metrics_collector
from agentic.guardrails.input_guard import JailbreakDetector, LengthGuard, PIIRedactor
from agentic.guardrails.output_guard import ToxicityFilter, OutputLengthGuard
from agentic.memory.short_term import ConversationBufferMemory
from agentic.patterns.reflection import ReflectionPattern
from agentic.patterns.parallelization import ParallelizationPattern
from agentic.tools.builtin import CalculatorTool, CodeExecutorTool


async def main() -> None:
    llm = OpenAIProvider(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    metrics = get_metrics_collector()
    bus = get_event_bus()

    # ── Observability ─────────────────────────────────────────────────
    events_log: list[str] = []

    def log_event(event: Event) -> None:
        events_log.append(f"[{event.name}] {list(event.payload.keys())}")

    bus.subscribe_all(log_event)

    print("=== Full Agentic Pipeline ===\n")

    # ── 1. Tool Agent with guardrails and memory ──────────────────────
    print("--- 1. Tool Agent with Guardrails + Memory ---")

    memory = ConversationBufferMemory(max_turns=10)
    agent = ToolAgent(
        llm=llm,
        agent_id="FullPipelineAgent",
        memory=memory,
        system_prompt=(
            "You are an expert AI assistant with coding and math capabilities. "
            "Always provide clear, accurate, and helpful responses."
        ),
    )

    # Add tools
    agent.add_tool(CalculatorTool())
    agent.add_tool(CodeExecutorTool())

    # Add guardrails
    agent.add_guardrail(JailbreakDetector(raise_on_detect=False))
    agent.add_guardrail(LengthGuard(min_length=5, max_length=10000))
    agent.add_guardrail(PIIRedactor(redact_in_output=True))
    agent.add_guardrail(ToxicityFilter(raise_on_detect=False))
    agent.add_guardrail(OutputLengthGuard(max_length=2000))

    with metrics.measure("run_001", "FullPipelineAgent"):
        result = await agent.run(
            "Write Python code to compute the first 15 prime numbers and calculate their sum."
        )

    print(f"Answer: {result.content[:500]}\n")
    print(f"Tokens: {result.total_tokens} | Elapsed: {result.elapsed_seconds:.2f}s\n")

    # ── 2. Multi-turn conversation (memory test) ──────────────────────
    print("--- 2. Multi-turn Conversation (memory) ---")
    r1 = await agent.run("My name is Alex and I love Python programming.")
    print(f"Turn 1: {r1.content[:150]}")
    r2 = await agent.run("What's my name and what do I love?")
    print(f"Turn 2 (should remember): {r2.content[:150]}\n")

    # ── 3. Reflection pattern ─────────────────────────────────────────
    print("--- 3. Reflection Pattern ---")
    reflection = ReflectionPattern(
        producer_llm=llm,
        critic_llm=llm,
        max_iterations=2,
        approval_threshold=7,
    )
    refl_result = await reflection.run(
        "Explain the concept of recursion with a clear, beginner-friendly example."
    )
    print(f"Reflected output: {refl_result.output[:300]}...")
    print(f"Iterations: {refl_result.metadata.get('iterations')} | Approved: {refl_result.metadata.get('approved')}\n")

    # ── 4. Parallelization ────────────────────────────────────────────
    print("--- 4. Parallelization Pattern ---")

    async def perspective_1(input: str, ctx: object) -> str:
        r = await llm.generate([
            __import__("agentic").Message.human(f"From a software engineering perspective: {input}")
        ])
        return r.message.content

    async def perspective_2(input: str, ctx: object) -> str:
        r = await llm.generate([
            __import__("agentic").Message.human(f"From a data science perspective: {input}")
        ])
        return r.message.content

    parallel = ParallelizationPattern(
        llm=llm,
        tasks=[perspective_1, perspective_2],
        aggregation="summarize",
    )
    par_result = await parallel.run("How should I structure a large Python project?")
    print(f"Parallel synthesis: {par_result.output[:300]}...\n")

    # ── 5. Evaluation ─────────────────────────────────────────────────
    print("--- 5. LLM-as-a-Judge Evaluation ---")
    evaluator = LLMJudgeEvaluator(judge_llm=llm)
    eval_result = await evaluator.evaluate(
        question="Explain what an API is.",
        response=result.content,
        criteria=["relevance", "completeness", "clarity"],
    )
    print(f"Overall score: {eval_result.overall_score:.2f}/1.0")
    print(f"Passed: {eval_result.passed}")
    for criterion, score in eval_result.scores.items():
        print(f"  {criterion}: {score.score:.2f}")
    print()

    # ── 6. Metrics summary ────────────────────────────────────────────
    print("--- 6. Metrics Summary ---")
    summary = metrics.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()

    # ── 7. Events captured ────────────────────────────────────────────
    print("--- 7. Events Captured ---")
    print(f"Total events: {len(events_log)}")
    for ev in events_log[:10]:
        print(f"  {ev}")
    if len(events_log) > 10:
        print(f"  ... and {len(events_log) - 10} more")


if __name__ == "__main__":
    asyncio.run(main())
