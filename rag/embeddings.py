"""Pluggable embeddings.

Priority order:
1. ``sentence-transformers`` (``all-MiniLM-L6-v2``) when installed and
   explicitly requested via ``EMBEDDING_BACKEND=sentence-transformers``.
2. TF-IDF (pure NumPy, no downloads) -- the default. This is what the
   offline demo and the tests use, so the project runs with no network
   and no model download.

The TF-IDF embedder is deliberately simple and transparent: tokenise on
word characters, drop a small stop-word list, compute IDF on the indexed
corpus, and L2-normalise every vector so cosine similarity is a dot
product. It is good enough for a small, focused corpus; it is not a
substitute for a dense model on a large corpus -- see README limitations.
"""
from __future__ import annotations

import math
import os
import re
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)?")

# Small, conservative stop-word list. Financial keywords (revenue, margin,
# guidance, growth, ...) are intentionally NOT stop-words.
STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "was", "were", "is", "are", "by", "at", "as", "it", "its", "that",
    "this", "from", "we", "our", "us", "be", "has", "have", "had", "than",
    "into", "per", "which", "also", "after", "before", "during", "over",
    "under", "between", "about", "up", "down", "out", "not", "no",
}


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens, with stop-words removed.

    Synonym expansion merges forms that mean the same thing in
    financial text, so a query saying "Q4" matches a document saying
    "fourth quarter", and "guidance" matches "guided".
    """
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(_SYNONYMS.get(token, ()))
    return expanded


_SYNONYMS: dict[str, tuple[str, ...]] = {
    "q1": ("first", "quarter"),
    "q2": ("second", "quarter"),
    "q3": ("third", "quarter"),
    "q4": ("fourth", "quarter"),
    "guidance": ("guided",),
    "guided": ("guidance",),
    "fy": ("fiscal", "year"),
}


class EmbeddingModel(Protocol):
    """Minimal embedding interface used by the store and pipeline."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(n_texts, dim)`` float32 array, rows L2-normalised."""
        ...


class TfidfEmbedder:
    """TF-IDF embedder fitted on the indexed corpus (NumPy only)."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0, dtype=np.float32)
        self.dim = 0
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        """Build the vocabulary and IDF weights from *texts*."""
        docs_tokens = [tokenize(t) for t in texts]
        vocab: dict[str, int] = {}
        for tokens in docs_tokens:
            for token in set(tokens):
                if token not in vocab:
                    vocab[token] = len(vocab)
        n_docs = max(len(texts), 1)
        df = np.zeros(len(vocab), dtype=np.float64)
        for tokens in docs_tokens:
            for token in set(tokens):
                df[vocab[token]] += 1
        # Smoothed IDF, as in scikit-learn.
        idf = np.log((1 + n_docs) / (1 + df)) + 1.0
        self.vocab = vocab
        self.idf = idf.astype(np.float32)
        self.dim = len(vocab)
        self._fitted = True
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fitted before embed() -- call fit() first")
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            if not tokens:
                continue
            counts: dict[int, float] = {}
            for token in tokens:
                idx = self.vocab.get(token)
                if idx is not None:
                    counts[idx] = counts.get(idx, 0.0) + 1.0
            for idx, count in counts.items():
                # log(1 + tf) * idf
                out[i, idx] = math.log1p(count) * float(self.idf[idx])
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)

    @property
    def backend(self) -> str:
        return "tfidf"


class SentenceTransformerEmbedder:
    """Dense embeddings via sentence-transformers, when installed.

    Loaded lazily so importing this module never requires the package.
    Raises ImportError with a clear message when it is unavailable.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it (pip install sentence-transformers) or use the TF-IDF fallback."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.model_name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    @property
    def backend(self) -> str:
        return f"sentence-transformers:{self.model_name}"


def get_embedder(prefer: str | None = None) -> EmbeddingModel:
    """Return an embedder.

    ``prefer`` (or env ``EMBEDDING_BACKEND``) may be
    ``"sentence-transformers"`` to try the dense model first; anything else
    -- including a failed dense load -- falls back to TF-IDF, which always
    works offline.
    """
    requested = (prefer or os.environ.get("EMBEDDING_BACKEND", "tfidf")).lower()
    if requested in {"sentence-transformers", "dense", "st"}:
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            pass  # fall through to the offline TF-IDF embedder
    return TfidfEmbedder()
