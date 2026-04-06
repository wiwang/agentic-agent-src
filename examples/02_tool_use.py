"""Example 02: Agent with tools (function calling).

Demonstrates: ToolAgent with Calculator, WebSearch, CodeExecutor.
Also shows the @tool decorator for quick function-based tools.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agentic import OpenAIProvider, ToolAgent, tool
from agentic.tools.builtin import CalculatorTool, WebSearchTool, CodeExecutorTool


@tool(description="Get the current date and time")
def get_current_time() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


async def main() -> None:
    llm = OpenAIProvider(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent = ToolAgent(
        llm=llm,
        agent_id="ToolAgent",
        system_prompt=(
            "You are a helpful assistant with access to tools. "
            "Use tools when needed to provide accurate answers."
        ),
    )

    # Register tools
    agent.add_tool(CalculatorTool())
    agent.add_tool(WebSearchTool(max_results=3))
    agent.add_tool(CodeExecutorTool())
    agent.add_tool(get_current_time)

    print("=== Tool Use Example ===\n")

    # Test 1: Calculator
    print("--- Test 1: Math calculation ---")
    result = await agent.run("Calculate: (2**32 + sqrt(144)) / pi. Use the calculator tool.")
    print(f"Answer: {result.content}\n")

    # Test 2: Code execution
    print("--- Test 2: Code execution ---")
    result = await agent.run(
        "Write and execute Python code to generate the first 10 Fibonacci numbers."
    )
    print(f"Answer: {result.content}\n")

    # Test 3: Current time
    print("--- Test 3: Current time ---")
    result = await agent.run("What time is it right now?")
    print(f"Answer: {result.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
