"""Custom exception hierarchy for the agentic framework."""

from __future__ import annotations


class AgentError(Exception):
    """Base exception for all agentic framework errors."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


# ── LLM / Provider errors ────────────────────────────────────────────────────


class LLMError(AgentError):
    """Raised when an LLM provider call fails."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider returns a rate-limit response."""


class LLMAuthError(LLMError):
    """Raised on authentication / API-key failures."""


class LLMContextLengthError(LLMError):
    """Raised when the prompt exceeds the model's context window."""


# ── Tool errors ───────────────────────────────────────────────────────────────


class ToolError(AgentError):
    """Raised when a tool execution fails."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not registered."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"Tool '{tool_name}' is not registered.")
        self.tool_name = tool_name


class ToolValidationError(ToolError):
    """Raised when tool input validation fails."""


# ── Memory errors ─────────────────────────────────────────────────────────────


class MemoryError(AgentError):  # noqa: A001  (shadows built-in intentionally)
    """Raised when a memory operation fails."""


# ── Pattern errors ────────────────────────────────────────────────────────────


class PatternError(AgentError):
    """Raised when a design pattern execution fails."""


class RoutingError(PatternError):
    """Raised when no route can be determined."""


class PlanningError(PatternError):
    """Raised during plan generation or execution."""


class ReflectionError(PatternError):
    """Raised during producer-critic reflection."""


# ── RAG errors ────────────────────────────────────────────────────────────────


class RAGError(AgentError):
    """Raised when a RAG pipeline operation fails."""


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""


class RetrievalError(RAGError):
    """Raised when vector retrieval fails."""


# ── Guardrail errors ──────────────────────────────────────────────────────────


class GuardrailError(AgentError):
    """Raised when a guardrail blocks execution."""


class InputGuardrailError(GuardrailError):
    """Raised when input validation fails or jailbreak is detected."""


class OutputGuardrailError(GuardrailError):
    """Raised when output filtering blocks a response."""


# ── Plugin errors ─────────────────────────────────────────────────────────────


class PluginError(AgentError):
    """Raised when plugin loading or execution fails."""


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin cannot be found."""


# ── HITL errors ───────────────────────────────────────────────────────────────


class HITLError(AgentError):
    """Raised when a human-in-the-loop checkpoint fails or is rejected."""


class HITLRejectionError(HITLError):
    """Raised when a human explicitly rejects a checkpoint."""


# ── Evaluation errors ─────────────────────────────────────────────────────────


class EvaluationError(AgentError):
    """Raised when evaluation fails."""


# ── A2A / MCP errors ──────────────────────────────────────────────────────────


class A2AError(AgentError):
    """Raised during agent-to-agent communication."""


class MCPError(AgentError):
    """Raised during Model Context Protocol operations."""
