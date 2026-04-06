"""Abstract base for RAG components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentic.rag.retriever import RetrievedChunk


class BaseRAG(ABC):
    """Minimal abstract interface for RAG implementations."""

    @abstractmethod
    async def add_text(self, text: str, source: str = "") -> int:
        """Index text. Returns number of chunks."""

    @abstractmethod
    async def query(self, question: str, **kwargs: Any) -> Any:
        """Answer a question using retrieved context."""
