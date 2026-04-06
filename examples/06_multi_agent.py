"""Example 06: Multi-Agent patterns.

Demonstrates: Supervisor, Sequential, Parallel, and Network topologies.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agentic import OpenAIProvider, ToolAgent
from agentic.patterns.multi_agent import (
    SupervisorPattern,
    SequentialPattern,
    ParallelAgentPattern,
)


async def main() -> None:
    llm = OpenAIProvider(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

    # Create specialist agents
    researcher = ToolAgent(
        llm=llm,
        agent_id="researcher",
        system_prompt="You are a research specialist. Provide factual, detailed information.",
    )
    analyst = ToolAgent(
        llm=llm,
        agent_id="analyst",
        system_prompt="You are a data analyst. Analyze information and identify patterns and insights.",
    )
    writer = ToolAgent(
        llm=llm,
        agent_id="writer",
        system_prompt="You are a technical writer. Transform complex information into clear, readable content.",
    )

    topic = "The environmental impact of AI data centers"
    print("=== Multi-Agent Example ===\n")
    print(f"Topic: {topic}\n")

    # --- Pattern 1: Sequential ---
    print("--- Sequential Pattern (Researcher → Analyst → Writer) ---")
    sequential = SequentialPattern(agents=[researcher, analyst, writer])
    result = await sequential.run(f"Research, analyze, and write about: {topic}")
    print(f"Output (first 400 chars): {result.output[:400]}...\n")

    # --- Pattern 2: Parallel ---
    print("--- Parallel Pattern (all agents on same input, then merge) ---")
    parallel = ParallelAgentPattern(
        agents=[researcher, analyst],
        aggregator_llm=llm,
    )
    result = await parallel.run(f"Provide your perspective on: {topic}")
    print(f"Output (first 400 chars): {result.output[:400]}...\n")

    # --- Pattern 3: Supervisor ---
    print("--- Supervisor Pattern ---")
    supervisor = SupervisorPattern(
        supervisor_llm=llm,
        agents={
            "researcher": researcher,
            "analyst": analyst,
            "writer": writer,
        },
    )
    result = await supervisor.run(
        f"Produce a comprehensive report on: {topic}"
    )
    print(f"Output (first 400 chars): {result.output[:400]}...")
    print(f"\nDelegations: {result.metadata.get('delegations')}")


if __name__ == "__main__":
    asyncio.run(main())
