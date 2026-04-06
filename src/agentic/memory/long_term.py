"""Long-term vector-store memory backed by ChromaDB."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from agentic.memory.base import BaseMemory

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


class VectorStoreMemory(BaseMemory):
    """Persistent long-term memory using ChromaDB for semantic retrieval.

    Requires chromadb and sentence-transformers (or openai embeddings).
    """

    def __init__(
        self,
        collection_name: str = "agentic_memory",
        persist_dir: str = "./.chroma",
        embedding_model: str = "all-MiniLM-L6-v2",
        use_openai_embeddings: bool = False,
        openai_api_key: str | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._persist_dir = persist_dir
        self._embedding_model = embedding_model
        self._use_openai = use_openai_embeddings
        self._openai_key = openai_api_key
        self._client: Any = None
        self._collection: Any = None
        self._ef: Any = None

    def _ensure_init(self) -> None:
        if self._client is not None:
            return
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as e:
            raise ImportError("Install 'chromadb' for VectorStoreMemory.") from e

        self._client = chromadb.PersistentClient(path=self._persist_dir)

        if self._use_openai:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
            self._ef = OpenAIEmbeddingFunction(
                api_key=self._openai_key, model_name="text-embedding-3-small"
            )
        else:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            self._ef = SentenceTransformerEmbeddingFunction(
                model_name=self._embedding_model
            )

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._ef,
        )

    async def store(
        self,
        input: str,
        output: str,
        context: "AgentContext | None" = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._ensure_init()
        text = f"User: {input}\nAssistant: {output}"
        doc_id = hashlib.md5(text.encode()).hexdigest()
        meta: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_preview": input[:200],
        }
        if metadata:
            meta.update({k: str(v) for k, v in metadata.items()})
        if context:
            meta["session_id"] = context.session_id

        self._collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[meta],
        )

    async def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        self._ensure_init()
        count = self._collection.count()
        if count == 0:
            return []
        n = min(top_k, count)
        results = self._collection.query(
            query_texts=[query],
            n_results=n,
        )
        docs = results.get("documents", [[]])[0]
        return docs

    async def clear(self) -> None:
        self._ensure_init()
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._ef,
        )

    async def get_all(self) -> list[dict[str, Any]]:
        self._ensure_init()
        results = self._collection.get()
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        return [{"document": d, "metadata": m} for d, m in zip(docs, metas)]
