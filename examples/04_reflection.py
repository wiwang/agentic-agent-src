"""Example 04: Reflection (Producer-Critic) pattern.

Demonstrates: ReflectionPattern iteratively improving a piece of writing.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agentic import OpenAIProvider
from agentic.patterns.reflection import ReflectionPattern


async def main() -> None:
    producer_llm = OpenAIProvider(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    # Using the same LLM as critic; in production you'd use a stronger model for the critic
    critic_llm = OpenAIProvider(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

    pattern = ReflectionPattern(
        producer_llm=producer_llm,
        critic_llm=critic_llm,
        max_iterations=3,
        approval_threshold=8,
    )

    print("=== Reflection (Producer-Critic) Example ===\n")

    request = (
        "Write a compelling 3-paragraph introduction for a blog post about "
        "why Python is the best language for AI development."
    )
    print(f"Request: {request}\n")

    result = await pattern.run(request)

    print("=== Final Output ===")
    print(result.output)
    print(f"\n--- Metadata ---")
    print(f"Iterations: {result.metadata.get('iterations')}")
    print(f"Approved: {result.metadata.get('approved')}")
    print(f"Final feedback: {result.metadata.get('final_feedback', '')[:200]}")
    print(f"\n--- Steps ---")
    for step in result.steps:
        print(step[:120])


if __name__ == "__main__":
    asyncio.run(main())
