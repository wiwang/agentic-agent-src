"""Example 05: Planning pattern (Plan-then-Execute).

Demonstrates: PlanningPattern generating and executing a multi-step plan.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agentic import OpenAIProvider
from agentic.patterns.planning import PlanningPattern
from agentic.tools.builtin import CalculatorTool, CodeExecutorTool


async def main() -> None:
    llm = OpenAIProvider(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

    planner = PlanningPattern(
        llm=llm,
        tools=[CalculatorTool(), CodeExecutorTool()],
    )

    print("=== Planning Pattern Example ===\n")

    goal = (
        "Analyze the performance characteristics of different sorting algorithms: "
        "bubble sort, merge sort, and quicksort. "
        "Write Python code to benchmark them on a list of 1000 random integers, "
        "then summarize the results."
    )
    print(f"Goal: {goal}\n")

    result = await planner.run(goal)

    print("=== Final Answer ===")
    print(result.output)
    print(f"\n--- Plan execution ---")
    print(f"Steps: {result.metadata.get('plan_steps')}")
    print(f"Completed: {result.metadata.get('completed_steps')}")
    print(f"Reasoning: {result.metadata.get('plan_reasoning', '')[:200]}")
    print("\n--- Step log ---")
    for step in result.steps[:5]:
        print(step[:150])


if __name__ == "__main__":
    asyncio.run(main())
