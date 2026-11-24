"""End-to-end RAG pipeline: ingest -> embed -> retrieve -> generate."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from rag.embeddings import EmbeddingModel, get_embedder
from rag.ingest import Chunk, Document, ingest_and_chunk
from rag.llm import LLMAnswer, get_llm
from rag.store import SearchResult, get_store

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"

# Cosine-similarity floor for a chunk to count as evidence. TF-IDF scores
# for a genuinely relevant chunk in this corpus are typically 0.15-0.5;
# unrelated queries score near 0. 0.08 separates the two cleanly here.
RELEVANCE_THRESHOLD = 0.08


@dataclass
class Citation:
    chunk_id: str
    doc_id: str
    company: str
    doc_type: str
    date: str
    source: str
    score: float
    text: str


@dataclass
class AskResult:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: str = "low"  # low | medium | high
    mode: str = "mock-extractive"
    refused: bool = False
    retrieved_chunks: list[Citation] = field(default_factory=list)


def _unsupported_entity_or_year(question: str, chunks: list[Chunk]) -> str | None:
    """Detect a named entity / year in *question* that no chunk mentions.

    TF-IDF cosine cannot see out-of-vocabulary tokens (e.g. "Tesla" is
    simply ignored, so "Tesla revenue" scores like "revenue" alone).
    This check closes that hole: a capitalised name other than the first
    word / interrogative, or a four-digit year, that appears in no
    retrieved chunk is grounds for refusal, not for answering from a
    different company's numbers.
    """
    corpus = " ".join(c.text for c in chunks).lower()
    # Capitalised words (not sentence-initial, not What/How/When/...).
    for word in re.findall(r"\b[A-Z][a-z]+\b", question):
        if word in {"What", "How", "When", "Why", "Which", "Who"}:
            continue
        if word.lower() not in corpus:
            return word
    for year in re.findall(r"\b(?:19|20)\d{2}\b", question):
        if year not in corpus:
            return year
    return None


class ResearchAgent:
    """A self-contained financial research agent over a local corpus."""

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        embedder: EmbeddingModel | None = None,
        llm=None,
        top_k: int = 4,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.top_k = top_k
        self.embedder = embedder or get_embedder()
        self.llm = llm or get_llm()
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.store = None
        self._load()

    def _load(self) -> None:
        self.documents, self.chunks = ingest_and_chunk(self.data_dir)
        if not self.chunks:
            self.store = get_store(1)
            return
        texts = [c.text for c in self.chunks]
        # TF-IDF embedders need fitting; dense embedders ignore fit().
        fit = getattr(self.embedder, "fit", None)
        if callable(fit):
            fit(texts)
        vectors = self.embedder.embed(texts)
        self.store = get_store(int(vectors.shape[1]))
        self.store.add(self.chunks, vectors)

    @property
    def backend(self) -> str:
        embedder_name = getattr(self.embedder, "backend", type(self.embedder).__name__)
        return f"embeddings={embedder_name} llm={getattr(self.llm, 'mode', 'unknown')}"

    def retrieve(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        """Return the top-k chunks for *question*, best first."""
        if not self.chunks or self.store is None:
            return []
        query_vec = self.embedder.embed([question])[0]
        return self.store.search(query_vec, top_k=top_k or self.top_k)

    def ask(self, question: str, top_k: int | None = None) -> AskResult:
        """Answer *question* from the corpus, with citations or a refusal."""
        question = question.strip()
        if not question:
            return AskResult(
                question=question,
                answer="Insufficient evidence in the provided documents.",
                confidence="low",
                mode=getattr(self.llm, "mode", "mock-extractive"),
                refused=True,
            )
        results = self.retrieve(question, top_k=top_k)
        relevant = [r for r in results if r.score >= RELEVANCE_THRESHOLD]
        retrieved = [
            Citation(
                chunk_id=r.chunk.chunk_id,
                doc_id=r.chunk.doc_id,
                company=r.chunk.company,
                doc_type=r.chunk.doc_type,
                date=r.chunk.date,
                source=r.chunk.source,
                score=round(r.score, 4),
                text=r.chunk.text,
            )
            for r in results
        ]
        unsupported = _unsupported_entity_or_year(question, [r.chunk for r in results]) if results else question
        if not relevant or unsupported:
            return AskResult(
                question=question,
                answer="Insufficient evidence in the provided documents.",
                confidence="low",
                mode=getattr(self.llm, "mode", "mock-extractive"),
                refused=True,
                retrieved_chunks=retrieved,
            )
        llm_answer: LLMAnswer = self.llm.generate(question, relevant)
        top_score = relevant[0].score if relevant else 0.0
        confidence = "high" if top_score >= 0.30 else "medium" if top_score >= 0.15 else "low"
        # Cite only the chunks the LLM actually used, not every retrieved
        # chunk: citing an unused chunk inflates citation counts and hurts
        # citation precision (and misleads the reader).
        used = set(llm_answer.used_chunks) or {relevant[0].chunk.chunk_id}
        citations = [
            Citation(
                chunk_id=r.chunk.chunk_id,
                doc_id=r.chunk.doc_id,
                company=r.chunk.company,
                doc_type=r.chunk.doc_type,
                date=r.chunk.date,
                source=r.chunk.source,
                score=round(r.score, 4),
                text=r.chunk.text,
            )
            for r in relevant
            if r.chunk.chunk_id in used
        ]
        return AskResult(
            question=question,
            answer=llm_answer.answer,
            citations=citations,
            confidence=confidence,
            mode=llm_answer.mode,
            refused=False,
            retrieved_chunks=retrieved,
        )


def get_default_agent() -> ResearchAgent:
    """Build an agent over the bundled sample docs (offline by default)."""
    return ResearchAgent(data_dir=os.environ.get("DATA_DIR", str(DEFAULT_DATA_DIR)))
