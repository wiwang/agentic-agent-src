"""Example 01: Basic agent with no tools.

Demonstrates: BaseAgent, ToolAgent, AgentContext, Message lifecycle.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agentic import OpenAIProvider, ToolAgent


async def main() -> None:
    # Create an LLM provider
    llm = OpenAIProvider(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
    )

    # Create a basic agent
    agent = ToolAgent(
        llm=llm,
        agent_id="BasicAgent",
        system_prompt="You are a helpful assistant. Be concise.",
    )

    print("=== Basic Agent Example ===\n")

    # Run the agent
    result = await agent.run("What is the capital of France? Answer in one sentence.")

    print(f"Answer: {result.content}")
    print(f"\nRun ID: {result.run_id}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Elapsed: {result.elapsed_seconds:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
