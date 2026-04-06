"""Agentic — A comprehensive framework implementing 21 agentic design patterns.

Quick start::

    from agentic import OpenAIProvider, ToolAgent
    from agentic.tools.builtin import CalculatorTool

    llm = OpenAIProvider(model="gpt-4o")
    agent = ToolAgent(llm=llm, system_prompt="You are a helpful assistant.")
    agent.add_tool(CalculatorTool())

    import asyncio
    result = asyncio.run(agent.run("What is 42 * 1337?"))
    print(result.content)
"""

from agentic.config import AgentConfig, get_config, set_config
from agentic.exceptions import AgentError

# Core
from agentic.core.agent import BaseAgent, AgentResponse
from agentic.core.context import AgentContext, AgentState
from agentic.core.message import Message, Role, ToolCall, ToolResult
from agentic.core.events import EventBus, get_event_bus

# LLM Providers
from agentic.llm.base import BaseLLMProvider, LLMResponse, ToolSchema
from agentic.llm.openai_provider import OpenAIProvider
from agentic.llm.anthropic_provider import AnthropicProvider

# Tools
from agentic.tools.base import BaseTool, FunctionTool, tool
from agentic.tools.registry import ToolRegistry, get_registry

# Memory
from agentic.memory.base import BaseMemory
from agentic.memory.short_term import ConversationBufferMemory
from agentic.memory.long_term import VectorStoreMemory

# Patterns
from agentic.patterns.base import BasePattern, PatternResult
from agentic.patterns.prompt_chaining import PromptChainingPattern
from agentic.patterns.routing import LLMRouter, RuleBasedRouter
from agentic.patterns.parallelization import ParallelizationPattern
from agentic.patterns.reflection import ReflectionPattern
from agentic.patterns.planning import PlanningPattern
from agentic.patterns.multi_agent import (
    SupervisorPattern,
    SequentialPattern,
    ParallelAgentPattern,
    NetworkPattern,
)

# Reasoning
from agentic.reasoning.chain_of_thought import ChainOfThoughtReasoner
from agentic.reasoning.tree_of_thought import TreeOfThoughtReasoner
from agentic.reasoning.react import ReActReasoner

# RAG
from agentic.rag.pipeline import RAGPipeline, AgenticRAGPipeline

# Guardrails
from agentic.guardrails.input_guard import JailbreakDetector, PIIRedactor
from agentic.guardrails.output_guard import ToxicityFilter

# Evaluation
from agentic.evaluation.evaluator import LLMJudgeEvaluator

# Plugins
from agentic.plugins.manager import PluginManager, get_plugin_manager

# Built-in agent implementations
from agentic.agents import ToolAgent, ReasoningAgent

__version__ = "0.1.0"

__all__ = [
    # Config
    "AgentConfig",
    "get_config",
    "set_config",
    "AgentError",
    # Core
    "BaseAgent",
    "AgentResponse",
    "AgentContext",
    "AgentState",
    "Message",
    "Role",
    "ToolCall",
    "ToolResult",
    "EventBus",
    "get_event_bus",
    # LLM
    "BaseLLMProvider",
    "LLMResponse",
    "ToolSchema",
    "OpenAIProvider",
    "AnthropicProvider",
    # Tools
    "BaseTool",
    "FunctionTool",
    "tool",
    "ToolRegistry",
    "get_registry",
    # Memory
    "BaseMemory",
    "ConversationBufferMemory",
    "VectorStoreMemory",
    # Patterns
    "BasePattern",
    "PatternResult",
    "PromptChainingPattern",
    "LLMRouter",
    "RuleBasedRouter",
    "ParallelizationPattern",
    "ReflectionPattern",
    "PlanningPattern",
    "SupervisorPattern",
    "SequentialPattern",
    "ParallelAgentPattern",
    "NetworkPattern",
    # Reasoning
    "ChainOfThoughtReasoner",
    "TreeOfThoughtReasoner",
    "ReActReasoner",
    # RAG
    "RAGPipeline",
    "AgenticRAGPipeline",
    # Guardrails
    "JailbreakDetector",
    "PIIRedactor",
    "ToxicityFilter",
    # Evaluation
    "LLMJudgeEvaluator",
    # Plugins
    "PluginManager",
    "get_plugin_manager",
    # Agents
    "ToolAgent",
    "ReasoningAgent",
    # Version
    "__version__",
]
