"""Example 03: Prompt Chaining pattern.

Demonstrates: PromptChainingPattern with sequential LLM steps.
Use case: Research → Summarize → Format as report.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agentic import OpenAIProvider
from agentic.patterns.prompt_chaining import PromptChainingPattern, ChainStep


async def main() -> None:
    llm = OpenAIProvider(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

    print("=== Prompt Chaining Example ===\n")

    # Build a 4-step research pipeline
    chain = PromptChainingPattern(llm=llm)
    chain.add_step(
        name="brainstorm",
        prompt_template=(
            "Brainstorm 5 key aspects of the following topic that are most important "
            "to understand. Format as a numbered list.\n\nTopic: {input}"
        ),
    )
    chain.add_step(
        name="research",
        prompt_template=(
            "For each of the following aspects, write 2-3 sentences explaining it clearly.\n\n"
            "Topic: {step_0}\n\n"
            "Original topic: {input}"
        ),
    )
    chain.add_step(
        name="critique",
        prompt_template=(
            "Review the following research and identify any gaps, misconceptions, "
            "or areas needing more depth:\n\n{previous}"
        ),
    )
    chain.add_step(
        name="final_report",
        prompt_template=(
            "Synthesize the research and critique into a concise, well-structured summary.\n\n"
            "Research: {step_1}\n\nCritique: {step_2}\n\nOriginal topic: {input}"
        ),
    )

    topic = "The impact of large language models on software development"
    print(f"Topic: {topic}\n")

    result = await chain.run(topic)

    print("=== Final Report ===")
    print(result.output)
    print(f"\nCompleted {result.metadata.get('n_steps')} steps")
    print(f"\n--- Step traces ---")
    for step in result.steps:
        print(step[:150] + "..." if len(step) > 150 else step)


if __name__ == "__main__":
    asyncio.run(main())
