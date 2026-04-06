"""Example 07: RAG (Retrieval-Augmented Generation) pipeline.

Demonstrates: RAGPipeline with text indexing and semantic question answering.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agentic import OpenAIProvider
from agentic.rag.pipeline import RAGPipeline, AgenticRAGPipeline
from agentic.rag.chunker import RecursiveChunker


SAMPLE_DOCUMENTS = [
    (
        "python_history.txt",
        """
        Python was created by Guido van Rossum and first released in 1991.
        The language was designed with emphasis on code readability and simplicity.
        Python 2.0 was released in 2000 and introduced garbage collection and Unicode support.
        Python 3.0, released in 2008, introduced many backward-incompatible changes to improve
        the language's consistency and reduce ambiguity.
        The Python Software Foundation (PSF) is the non-profit organization that governs Python development.
        As of 2024, Python consistently ranks as one of the top 3 most popular programming languages.
        """,
    ),
    (
        "python_features.txt",
        """
        Python supports multiple programming paradigms including procedural, object-oriented,
        and functional programming.
        Key features include dynamic typing, automatic memory management, and an extensive
        standard library often called "batteries included."
        Python's package ecosystem (PyPI) contains over 400,000 packages as of 2024.
        The Global Interpreter Lock (GIL) in CPython limits true multi-threading but async/await
        provides excellent concurrency for I/O-bound tasks.
        Type hints were introduced in Python 3.5 (PEP 484) for optional static typing.
        """,
    ),
    (
        "python_ai.txt",
        """
        Python has become the dominant language for artificial intelligence and machine learning.
        Major AI frameworks like TensorFlow, PyTorch, JAX, and scikit-learn all have Python as
        their primary interface.
        The language's simplicity makes it ideal for rapid prototyping of AI systems.
        Libraries like NumPy, Pandas, and Matplotlib form the data science stack.
        Hugging Face's Transformers library has made deploying state-of-the-art NLP models
        accessible to millions of Python developers.
        LangChain and similar frameworks enable building applications with large language models.
        """,
    ),
]


async def main() -> None:
    llm = OpenAIProvider(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

    print("=== RAG Pipeline Example ===\n")

    # Create RAG pipeline with local (sentence-transformer) embeddings
    pipeline = RAGPipeline(
        llm=llm,
        chunker=RecursiveChunker(chunk_size=200, overlap=30),
        top_k=4,
    )

    # Index documents
    print("Indexing documents...")
    for filename, content in SAMPLE_DOCUMENTS:
        n = await pipeline.add_text(content, source=filename)
        print(f"  ✓ {filename}: {n} chunks")

    total = await pipeline.chunk_count()
    print(f"Total chunks indexed: {total}\n")

    # Query
    questions = [
        "When was Python created and by whom?",
        "What makes Python popular for AI development?",
        "What is the Global Interpreter Lock?",
        "How many packages are on PyPI?",
    ]

    for question in questions:
        print(f"Q: {question}")
        result = await pipeline.query(question)
        print(f"A: {result.answer}")
        print(f"   Sources: {', '.join(result.sources)}")
        print(f"   Retrieved {result.metadata.get('retrieved')} chunks\n")

    # Agentic RAG (with iterative query refinement)
    print("=== Agentic RAG (iterative retrieval) ===\n")
    agentic_pipeline = AgenticRAGPipeline(llm=llm, max_iterations=2)
    for filename, content in SAMPLE_DOCUMENTS:
        await agentic_pipeline.add_text(content, source=filename)

    result = await agentic_pipeline.query(
        "Compare Python's AI ecosystem to other languages and explain why it dominates."
    )
    print(f"Q: Compare Python's AI ecosystem...")
    print(f"A: {result.answer}")
    print(f"   Iterations: {result.metadata.get('iterations')}")
    print(f"   Sources: {', '.join(result.sources)}")


if __name__ == "__main__":
    asyncio.run(main())
