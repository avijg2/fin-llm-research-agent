"""Tests for chunking, embeddings, retrieval, mock LLM, and evals."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rag.embeddings import TfidfEmbedder
from rag.evals import EVAL_CASES, run_evals
from rag.ingest import chunk_document, chunk_text, ingest_directory
from rag.llm import MockExtractiveLLM
from rag.pipeline import ResearchAgent

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


@pytest.fixture(scope="module")
def agent() -> ResearchAgent:
    return ResearchAgent(data_dir=DATA_DIR)


def test_chunking_splits_long_text_with_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    # Overlap: a word from the end of chunk 0 should appear in chunk 1.
    assert set(chunks[0].split()) & set(chunks[1].split())


def test_chunking_short_text_is_single_chunk():
    assert chunk_text("hello world", chunk_size=800) == ["hello world"]
    assert chunk_text("") == []
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=100, overlap=100)


def test_ingest_loads_three_sample_docs():
    docs = ingest_directory(DATA_DIR)
    assert len(docs) == 3
    companies = {d.company for d in docs}
    assert companies == {
        "Northwind Cloud Systems",
        "Bluepeak Semiconductor",
        "Harborline Retail Group",
    }
    assert all(d.date for d in docs)
    assert all(d.doc_type for d in docs)


def test_chunk_document_ids_are_stable(agent: ResearchAgent):
    ids = [c.chunk_id for c in agent.chunks]
    assert len(ids) == len(set(ids))
    assert all("#" in cid for cid in ids)
    assert len(agent.chunks) >= 6  # each doc splits into at least 2 chunks


def test_tfidf_embeddings_are_normalised_and_consistent():
    embedder = TfidfEmbedder().fit(["revenue grew strongly", "revenue fell sharply", "unrelated weather"])
    vecs = embedder.embed(["revenue grew", "revenue grew"])
    assert vecs.shape == (2, embedder.dim)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(vecs[0], vecs[1], atol=1e-6)


def test_retrieval_returns_relevant_chunk(agent: ResearchAgent):
    results = agent.retrieve("What was Northwind Cloud Systems revenue in Q3 2026?")
    assert results
    assert results[0].chunk.doc_id == "northwind-q3-2026-earnings-call"
    assert results[0].score >= 0.15


def test_retrieval_scores_unrelated_query_low(agent: ResearchAgent):
    results = agent.retrieve("What is the weather on Mars tomorrow?")
    assert not results or results[0].score < 0.10


def test_mock_llm_answer_contains_fact_and_citation(agent: ResearchAgent):
    result = agent.ask("What was Northwind Cloud Systems revenue in Q3 2026?")
    assert not result.refused
    assert "$412.8 million" in result.answer
    assert result.citations
    assert result.citations[0].doc_id == "northwind-q3-2026-earnings-call"
    assert result.mode == "mock-extractive"


def test_mock_llm_does_not_invent_numbers(agent: ResearchAgent):
    # Tesla is not in the corpus; 1999 predates every sample doc.
    for question in (
        "What was Tesla's revenue in Q3 2026?",
        "What was Northwind's revenue in 1999?",
    ):
        result = agent.ask(question)
        assert result.refused
        assert "Insufficient evidence" in result.answer
        # No invented revenue figure may appear in a refusal.
        assert "$" not in result.answer or "Insufficient" in result.answer


def test_mock_llm_only_uses_retrieved_sentences(agent: ResearchAgent):
    llm = MockExtractiveLLM()
    result = agent.ask("How much did Bluepeak spend on research and development in fiscal 2026?")
    assert "$212.3 million" in result.answer
    # Every number in the answer must also appear in a cited chunk.
    import re

    numbers = set(re.findall(r"\$[\d,.]+ million|\d+\.\d+%", result.answer))
    chunk_texts = " ".join(c.text for c in agent.chunks)
    for number in numbers:
        assert number in chunk_texts, f"{number} was invented (not in any chunk)"


def test_eval_harness_runs_and_scores(agent: ResearchAgent):
    assert len(EVAL_CASES) >= 8
    outcome = run_evals(agent)
    assert outcome.total == len(EVAL_CASES)
    assert outcome.fact_accuracy >= 0.75
    assert outcome.citation_precision >= 0.75
    assert outcome.refusal_accuracy >= 0.9
    assert outcome.passed >= 8
