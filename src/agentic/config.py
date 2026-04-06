"""Global configuration using pydantic-settings."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogFormat(str, Enum):
    JSON = "json"
    CONSOLE = "console"


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    LOCAL = "local"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class AgentConfig(BaseSettings):
    """Central configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_prefix="AGENTIC_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API Keys (no prefix — standard names) ─────────────────────────────
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # ── LLM Defaults ──────────────────────────────────────────────────────
    default_provider: LLMProvider = LLMProvider.OPENAI
    default_model: str = "gpt-4o"
    anthropic_model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    temperature: float = 0.7

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: LogFormat = LogFormat.CONSOLE

    # ── Plugins ───────────────────────────────────────────────────────────
    plugins_dir: str | None = None

    # ── RAG / Vector store ────────────────────────────────────────────────
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_persist_dir: str = "./.chroma"

    # ── Embeddings ────────────────────────────────────────────────────────
    embedding_provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Guardrails ────────────────────────────────────────────────────────
    guardrails_enabled: bool = True
    max_retries: int = 3

    # ── Evaluation ────────────────────────────────────────────────────────
    eval_judge_model: str = "gpt-4o"

    # ── Agent behaviour ───────────────────────────────────────────────────
    max_iterations: int = 20
    tool_timeout_seconds: float = 30.0


# Singleton-style default config; importers may instantiate their own.
_default_config: AgentConfig | None = None


def get_config() -> AgentConfig:
    global _default_config
    if _default_config is None:
        _default_config = AgentConfig()
    return _default_config


def set_config(config: AgentConfig) -> None:
    global _default_config
    _default_config = config
