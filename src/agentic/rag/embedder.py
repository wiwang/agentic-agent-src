"""Embedding providers for RAG pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEmbedder(ABC):
    """Abstract interface for generating text embeddings."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return a list of embedding vectors for the input texts."""

    async def embed_one(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]


class LocalEmbedder(BaseEmbedder):
    """Embeddings using sentence-transformers (runs locally, no API key)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any = None

    def _load(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError as e:
                raise ImportError(
                    "Install 'sentence-transformers' for LocalEmbedder."
                ) from e

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        self._load()
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(
            None, lambda: self._model.encode(texts, convert_to_numpy=True).tolist()
        )
        return vectors


class OpenAIEmbedder(BaseEmbedder):
    """Embeddings using OpenAI's text-embedding models."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 100,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=api_key)
        except ImportError as e:
            raise ImportError("Install 'openai' for OpenAIEmbedder.") from e

    async def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            resp = await self._client.embeddings.create(input=batch, model=self.model)
            all_embeddings.extend([d.embedding for d in resp.data])
        return all_embeddings
