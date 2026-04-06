"""Agentic design patterns."""

from agentic.patterns.base import BasePattern, PatternResult
from agentic.patterns.prompt_chaining import PromptChainingPattern, ChainStep
from agentic.patterns.routing import LLMRouter, RuleBasedRouter, Route
from agentic.patterns.parallelization import ParallelizationPattern, MapReducePattern
from agentic.patterns.reflection import ReflectionPattern
from agentic.patterns.planning import PlanningPattern, Plan, PlanStep
from agentic.patterns.multi_agent import (
    SupervisorPattern,
    SequentialPattern,
    ParallelAgentPattern,
    NetworkPattern,
)

__all__ = [
    "BasePattern",
    "PatternResult",
    "PromptChainingPattern",
    "ChainStep",
    "LLMRouter",
    "RuleBasedRouter",
    "Route",
    "ParallelizationPattern",
    "MapReducePattern",
    "ReflectionPattern",
    "PlanningPattern",
    "Plan",
    "PlanStep",
    "SupervisorPattern",
    "SequentialPattern",
    "ParallelAgentPattern",
    "NetworkPattern",
]
