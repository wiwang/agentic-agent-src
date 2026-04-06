"""Tests for memory subsystem."""

from __future__ import annotations

import pytest


# ── ConversationBufferMemory ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_buffer_memory_store_and_retrieve() -> None:
    from agentic.memory.short_term import ConversationBufferMemory

    memory = ConversationBufferMemory(max_turns=5)
    await memory.store("What is Python?", "Python is a programming language.")
    await memory.store("Who created it?", "Guido van Rossum created Python.")

    results = await memory.retrieve("Python creator", top_k=5)
    assert len(results) == 2
    assert any("Guido" in r for r in results)


@pytest.mark.asyncio
async def test_buffer_memory_max_turns() -> None:
    from agentic.memory.short_term import ConversationBufferMemory

    memory = ConversationBufferMemory(max_turns=3)
    for i in range(5):
        await memory.store(f"input {i}", f"output {i}")

    assert memory.turn_count == 3  # Ring buffer trims oldest
    results = await memory.retrieve("anything", top_k=10)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_buffer_memory_clear() -> None:
    from agentic.memory.short_term import ConversationBufferMemory

    memory = ConversationBufferMemory()
    await memory.store("hello", "world")
    await memory.clear()
    results = await memory.retrieve("hello")
    assert results == []


# ── SlidingWindowMemory ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sliding_window_respects_token_budget() -> None:
    from agentic.memory.short_term import SlidingWindowMemory

    memory = SlidingWindowMemory(max_tokens=50)  # ~200 chars
    # Store many large turns
    for i in range(20):
        await memory.store("x" * 50, "y" * 50)

    results = await memory.retrieve("anything")
    # Should have been trimmed
    assert len(results) < 20


# ── EpisodicMemory ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_episodic_memory_add_and_retrieve() -> None:
    from agentic.memory.episodic import EpisodicMemory

    memory = EpisodicMemory()
    ep = await memory.add_episode(
        task="Summarize a document",
        approach="Used RAG pipeline",
        outcome="Produced accurate summary",
        success=True,
        tags=["rag", "summarization"],
    )
    assert ep.id != ""

    results = await memory.retrieve("document summarization", top_k=3)
    assert len(results) >= 1
    assert "Summarize a document" in results[0]


@pytest.mark.asyncio
async def test_episodic_memory_filter() -> None:
    from agentic.memory.episodic import EpisodicMemory

    memory = EpisodicMemory()
    await memory.add_episode("task1", "approach1", "success", success=True)
    await memory.add_episode("task2", "approach2", "failure", success=False)

    assert len(memory.successful_episodes()) == 1
    assert len(memory.failed_episodes()) == 1


@pytest.mark.asyncio
async def test_episodic_memory_store_interface() -> None:
    from agentic.memory.episodic import EpisodicMemory

    memory = EpisodicMemory()
    await memory.store("user input", "agent output")
    results = await memory.retrieve("user")
    assert len(results) == 1


# ── Chunker tests ──────────────────────────────────────────────────────────────

def test_fixed_size_chunker() -> None:
    from agentic.rag.chunker import FixedSizeChunker

    chunker = FixedSizeChunker(chunk_size=50, overlap=10)
    text = "A" * 120
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    assert all(len(c.text) <= 50 for c in chunks)


def test_recursive_chunker() -> None:
    from agentic.rag.chunker import RecursiveChunker

    chunker = RecursiveChunker(chunk_size=100, overlap=20)
    text = "This is sentence one. This is sentence two.\n\nThis is a new paragraph."
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    assert all(len(c.text) <= 100 for c in chunks)


def test_sentence_chunker() -> None:
    from agentic.rag.chunker import SentenceChunker

    chunker = SentenceChunker(max_sentences=2, overlap_sentences=0)
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
