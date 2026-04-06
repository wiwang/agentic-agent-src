"""Vector retrieval using ChromaDB."""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel

from agentic.rag.chunker import Chunk


class RetrievedChunk(BaseModel):
    """A retrieved chunk with its similarity score."""

    text: str
    score: float
    metadata: dict[str, Any] = {}
    chunk_id: str = ""


class VectorRetriever:
    """Stores chunks in ChromaDB and retrieves by semantic similarity.

    Usage::

        retriever = VectorRetriever(collection_name="my_docs")
        await retriever.add_chunks(chunks)
        results = await retriever.retrieve("my query", top_k=5)
    """

    def __init__(
        self,
        collection_name: str = "agentic_rag",
        persist_dir: str = "./.chroma",
        embedding_model: str = "all-MiniLM-L6-v2",
        use_openai: bool = False,
        openai_api_key: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key
        self._client: Any = None
        self._collection: Any = None

    def _init(self) -> None:
        if self._client is not None:
            return
        try:
            import chromadb
        except ImportError as e:
            raise ImportError("Install 'chromadb' for VectorRetriever.") from e

        self._client = chromadb.PersistentClient(path=self.persist_dir)

        if self.use_openai:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
            ef = OpenAIEmbeddingFunction(
                api_key=self.openai_api_key, model_name="text-embedding-3-small"
            )
        else:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            ef = SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, embedding_function=ef
        )

    async def add_chunks(self, chunks: list[Chunk], source: str = "") -> None:
        self._init()
        if not chunks:
            return
        ids = [hashlib.md5(f"{source}{c.index}{c.text}".encode()).hexdigest() for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [{**c.metadata, "source": source, "index": c.index} for c in chunks]
        self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    async def add_text(self, text: str, source: str = "", chunk_size: int = 512) -> int:
        from agentic.rag.chunker import RecursiveChunker
        chunker = RecursiveChunker(chunk_size=chunk_size)
        chunks = chunker.chunk(text, metadata={"source": source})
        await self.add_chunks(chunks, source=source)
        return len(chunks)

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        self._init()
        count = self._collection.count()
        if count == 0:
            return []
        n = min(top_k, count)
        results = self._collection.query(query_texts=[query], n_results=n, include=["documents", "metadatas", "distances"])
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved = []
        for doc, meta, dist in zip(docs, metas, distances):
            # ChromaDB distance → similarity (lower distance = more similar)
            score = 1.0 - (dist / 2.0) if dist is not None else 0.5
            retrieved.append(
                RetrievedChunk(text=doc, score=score, metadata=meta or {})
            )
        return retrieved

    async def delete_source(self, source: str) -> None:
        self._init()
        results = self._collection.get(where={"source": source})
        ids = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)

    async def count(self) -> int:
        self._init()
        return self._collection.count()

    async def clear(self) -> None:
        self._init()
        self._client.delete_collection(self.collection_name)
        self._collection = None
        self._client = None
