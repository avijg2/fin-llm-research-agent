"""RAG package: ingestion, embeddings, vector store, LLM, and evals."""

from rag.ingest import Chunk, Document, chunk_text, ingest_directory
from rag.pipeline import ResearchAgent, get_default_agent

__all__ = [
    "Chunk",
    "Document",
    "ResearchAgent",
    "chunk_text",
    "get_default_agent",
    "ingest_directory",
]
