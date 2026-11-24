"""Vector store.

Priority order:
1. FAISS (``IndexFlatIP`` on L2-normalised vectors = cosine similarity)
   when ``faiss`` is installed and ``VECTOR_BACKEND=faiss`` is set.
2. NumPy in-memory store (default). Cosine similarity via dot product on
   normalised vectors. Exact, dependency-free, and fast enough for the
   small corpora this project targets.

scikit-learn is *not* required: if it happens to be installed it is not
used here, because the NumPy path is exact and has no extra dependency.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from rag.ingest import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity, in [-1, 1]


class NumpyVectorStore:
    """Exact cosine-similarity search over in-memory NumPy vectors."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors.size == 0:
            self.vectors = vectors
        else:
            if vectors.shape[1] != self.vectors.shape[1]:
                raise ValueError("vector dimension mismatch with existing store")
            self.vectors = np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[SearchResult]:
        if not self.chunks:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(q))
        if norm == 0:
            return []
        q = q / norm
        scores = self.vectors @ q  # rows are already L2-normalised
        order = np.argsort(-scores)[: max(top_k, 0)]
        return [SearchResult(chunk=self.chunks[int(i)], score=float(scores[int(i)])) for i in order]

    def __len__(self) -> int:
        return len(self.chunks)


class FaissVectorStore:
    """FAISS-backed store, used only when faiss is installed."""

    def __init__(self, dim: int) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("faiss is not installed; use NumpyVectorStore instead") from exc
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []
        self.dim = dim

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        self._index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[SearchResult]:
        if not self.chunks:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        norm = float(np.linalg.norm(q))
        if norm == 0:
            return []
        q = q / norm
        scores, indices = self._index.search(q, min(top_k, len(self.chunks)))
        return [
            SearchResult(chunk=self.chunks[int(i)], score=float(s))
            for s, i in zip(scores[0], indices[0])
            if i >= 0
        ]

    def __len__(self) -> int:
        return len(self.chunks)


def get_store(dim: int, prefer: str | None = None):
    """Return a vector store, preferring FAISS only when requested."""
    requested = (prefer or os.environ.get("VECTOR_BACKEND", "numpy")).lower()
    if requested == "faiss":
        try:
            return FaissVectorStore(dim)
        except Exception:
            pass  # fall through to NumPy
    return NumpyVectorStore()
