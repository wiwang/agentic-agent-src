# Agentic Framework

A comprehensive Python framework implementing **21 agentic design patterns** from the projects I participated in past 2 years.
They can be widely reused in the future work, I will keep adding new patterns I come up with and use in the promising projects

## Features

- **21 Design Patterns**: Prompt Chaining, Routing, Parallelization, Reflection, Tool Use, Planning, Multi-Agent, Memory, RAG, Guardrails, Evaluation, HITL, MCP, A2A, and more
- **Plugin System**: Extend with custom tools, memory, and guardrails via `entry_points`
- **Async-First**: All LLM calls use `asyncio` for non-blocking I/O
- **Multiple LLM Providers**: OpenAI and Anthropic Claude
- **Type-Safe**: Pydantic v2 throughout

## Quick Start

```bash
uv sync
cp .env.example .env  # Add your API keys
uv run python examples/01_basic_agent.py
```

## Running Examples

```bash
uv run python examples/01_basic_agent.py    # Basic agent
uv run python examples/02_tool_use.py       # Tool use (function calling)
uv run python examples/03_prompt_chaining.py # Prompt chaining
uv run python examples/04_reflection.py     # Reflection/Producer-Critic
uv run python examples/05_planning.py       # Plan-then-Execute
uv run python examples/06_multi_agent.py    # Multi-agent coordination
uv run python examples/07_rag_agent.py      # RAG pipeline
uv run python examples/08_full_pipeline.py  # Full pipeline
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Architecture

```
src/agentic/
├── core/          # BaseAgent, AgentContext, Message, EventBus
├── llm/           # OpenAI and Anthropic providers
├── tools/         # BaseTool, @tool decorator, ToolRegistry, built-ins
├── memory/        # Short-term, long-term (vector), episodic
├── patterns/      # All 21 design patterns
├── reasoning/     # CoT, ToT, ReAct
├── rag/           # Chunker, Embedder, VectorRetriever, RAGPipeline
├── guardrails/    # Input/output validation and safety
├── evaluation/    # LLM-as-a-Judge, metrics
├── hitl/          # Human-in-the-Loop checkpoints
├── mcp/           # Model Context Protocol client/server
├── a2a/           # Agent-to-Agent communication
└── plugins/       # Plugin system with entry_points discovery
```
