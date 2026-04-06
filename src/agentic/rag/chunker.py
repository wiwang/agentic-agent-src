"""Text chunking strategies for RAG pipelines."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class Chunk(BaseModel):
    """A text chunk with positional metadata."""

    text: str
    index: int
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = {}


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split text into chunks."""


class FixedSizeChunker(BaseChunker):
    """Split text into fixed-size character chunks with optional overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=idx,
                    start_char=start,
                    end_char=min(end, len(text)),
                    metadata=metadata or {},
                )
            )
            start += self.chunk_size - self.overlap
            idx += 1
        return chunks


class SentenceChunker(BaseChunker):
    """Chunk text by sentences, respecting a max token budget."""

    def __init__(self, max_sentences: int = 5, overlap_sentences: int = 1) -> None:
        self.max_sentences = max_sentences
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        sentences = self._split_sentences(text)
        chunks = []
        idx = 0
        i = 0
        while i < len(sentences):
            window = sentences[i: i + self.max_sentences]
            chunk_text = " ".join(window)
            chunks.append(Chunk(text=chunk_text, index=idx, metadata=metadata or {}))
            i += self.max_sentences - self.overlap_sentences
            idx += 1
        return chunks


class RecursiveChunker(BaseChunker):
    """Recursively splits on paragraph → sentence → word boundaries."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]
        sep = separators[0]
        parts = text.split(sep) if sep else list(text)
        good: list[str] = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    good.append(current)
                if len(part) <= self.chunk_size:
                    current = part
                else:
                    # Recurse with finer separator
                    sub = self._split(part, separators[1:])
                    good.extend(sub[:-1])
                    current = sub[-1] if sub else ""
        if current:
            good.append(current)
        return good

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        splits = self._split(text, self._separators)
        return [
            Chunk(text=s, index=i, metadata=metadata or {})
            for i, s in enumerate(splits)
            if s.strip()
        ]


class SemanticChunker(BaseChunker):
    """Group sentences with high embedding similarity into chunks.

    Falls back to SentenceChunker if embeddings are unavailable.
    """

    def __init__(
        self,
        embedder: Any | None = None,
        similarity_threshold: float = 0.8,
        max_chunk_size: int = 512,
    ) -> None:
        self._embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self._fallback = SentenceChunker(max_sentences=4)

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        if self._embedder is None:
            return self._fallback.chunk(text, metadata)

        import numpy as np

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return [Chunk(text=text, index=0, metadata=metadata or {})]

        # Compute embeddings synchronously (blocking)
        embeddings = self._embedder.encode(sentences)

        chunks: list[Chunk] = []
        current_sents: list[str] = [sentences[0]]
        idx = 0

        for i in range(1, len(sentences)):
            sim = float(np.dot(embeddings[i - 1], embeddings[i]) /
                        (np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i]) + 1e-9))
            current_text = " ".join(current_sents + [sentences[i]])
            if sim >= self.similarity_threshold and len(current_text) <= self.max_chunk_size:
                current_sents.append(sentences[i])
            else:
                chunks.append(Chunk(text=" ".join(current_sents), index=idx, metadata=metadata or {}))
                current_sents = [sentences[i]]
                idx += 1

        if current_sents:
            chunks.append(Chunk(text=" ".join(current_sents), index=idx, metadata=metadata or {}))

        return chunks
