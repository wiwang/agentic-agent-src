"""RAG Pipeline — orchestrates chunking, indexing, retrieval and generation."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

from agentic.core.message import Message
from agentic.rag.chunker import BaseChunker, RecursiveChunker
from agentic.rag.retriever import RetrievedChunk, VectorRetriever

if TYPE_CHECKING:
    from agentic.llm.base import BaseLLMProvider


class RAGResult(BaseModel):
    answer: str
    retrieved_chunks: list[RetrievedChunk] = []
    sources: list[str] = []
    metadata: dict[str, Any] = {}


RAG_SYSTEM_PROMPT = """You are a helpful assistant. Answer questions based ONLY on the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."
Always cite the source when possible."""

RAG_USER_PROMPT = """Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline.

    Workflow:
    1. Index documents via ``add_document()`` / ``add_text()``
    2. Answer questions via ``query()``

    Supports:
    - Custom chunkers
    - Custom retrievers (VectorRetriever by default)
    - Reranking (via score threshold)
    - Citation of sources
    """

    def __init__(
        self,
        llm: "BaseLLMProvider",
        retriever: VectorRetriever | None = None,
        chunker: BaseChunker | None = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
        system_prompt: str = RAG_SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm
        self.retriever = retriever or VectorRetriever()
        self.chunker = chunker or RecursiveChunker(chunk_size=512, overlap=64)
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.system_prompt = system_prompt

    async def add_text(self, text: str, source: str = "unknown") -> int:
        """Chunk and index raw text. Returns number of chunks added."""
        chunks = self.chunker.chunk(text, metadata={"source": source})
        await self.retriever.add_chunks(chunks, source=source)
        return len(chunks)

    async def add_document(self, path: str) -> int:
        """Read a text file and index its contents."""
        from pathlib import Path
        content = Path(path).read_text(encoding="utf-8")
        return await self.add_text(content, source=path)

    async def query(self, question: str, **kwargs: Any) -> RAGResult:
        """Retrieve relevant chunks and generate an answer."""
        retrieved = await self.retriever.retrieve(question, top_k=self.top_k)

        # Filter by score threshold
        if self.score_threshold > 0:
            retrieved = [r for r in retrieved if r.score >= self.score_threshold]

        if not retrieved:
            return RAGResult(
                answer="I don't have enough information to answer that question.",
                retrieved_chunks=[],
                metadata={"retrieved": 0},
            )

        # Build context
        context_parts = []
        sources = []
        for i, chunk in enumerate(retrieved):
            src = chunk.metadata.get("source", "unknown")
            context_parts.append(f"[Source {i+1}: {src}]\n{chunk.text}")
            if src not in sources:
                sources.append(src)

        context = "\n\n".join(context_parts)
        user_prompt = RAG_USER_PROMPT.format(context=context, question=question)

        messages = [
            Message.system(self.system_prompt),
            Message.human(user_prompt),
        ]
        response = await self.llm.generate(messages)

        return RAGResult(
            answer=response.message.content,
            retrieved_chunks=retrieved,
            sources=sources,
            metadata={
                "retrieved": len(retrieved),
                "model": response.model,
            },
        )

    async def chunk_count(self) -> int:
        return await self.retriever.count()


class AgenticRAGPipeline(RAGPipeline):
    """Agentic RAG: iteratively refines queries for better retrieval.

    If the initial answer is uncertain, the LLM generates follow-up queries
    and retrieves additional context before producing a final answer.
    """

    def __init__(self, *args: Any, max_iterations: int = 2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_iterations = max_iterations

    async def query(self, question: str, **kwargs: Any) -> RAGResult:
        all_chunks: list[RetrievedChunk] = []
        current_question = question

        for i in range(self.max_iterations):
            retrieved = await self.retriever.retrieve(current_question, top_k=self.top_k)
            all_chunks.extend(retrieved)

            # Deduplicate by text
            seen = set()
            unique_chunks = []
            for c in all_chunks:
                if c.text not in seen:
                    seen.add(c.text)
                    unique_chunks.append(c)
            all_chunks = unique_chunks

            # Check if we need more context
            if i < self.max_iterations - 1:
                context = "\n\n".join(c.text for c in all_chunks[:self.top_k])
                refine_prompt = (
                    f"Given this context:\n{context}\n\n"
                    f"For question: '{question}'\n\n"
                    f"Is the context sufficient? If not, what specific follow-up search query would help? "
                    f"Respond with 'SUFFICIENT' or 'SEARCH: <query>'"
                )
                resp = await self.llm.generate([Message.human(refine_prompt)])
                raw = resp.message.content.strip()
                if raw.upper().startswith("SEARCH:"):
                    current_question = raw[7:].strip()
                else:
                    break

        # Generate final answer with all accumulated chunks
        context = "\n\n".join(
            f"[{i+1}] {c.text}" for i, c in enumerate(all_chunks[:self.top_k * 2])
        )
        user_prompt = RAG_USER_PROMPT.format(context=context, question=question)
        messages = [Message.system(self.system_prompt), Message.human(user_prompt)]
        response = await self.llm.generate(messages)

        sources = list({c.metadata.get("source", "unknown") for c in all_chunks})
        return RAGResult(
            answer=response.message.content,
            retrieved_chunks=all_chunks,
            sources=sources,
            metadata={"retrieved": len(all_chunks), "iterations": i + 1},
        )
