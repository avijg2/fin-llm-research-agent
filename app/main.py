"""FastAPI service for the financial research agent.

Endpoints:
    GET  /health  -- liveness check (does not need the corpus loaded)
    POST /ask     -- answer a question from the indexed documents
    POST /ingest  -- re-ingest the bundled sample documents
    GET  /docs    -- interactive API docs (built in by FastAPI)

Run locally (no API key needed, offline mock mode):
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.pipeline import AskResult, ResearchAgent, get_default_agent

app = FastAPI(
    title="Fin LLM Research Agent",
    description=(
        "Retrieval-augmented financial research over earnings calls and "
        "filings. Answers are grounded in retrieved chunks and carry "
        "citations; out-of-scope questions are refused, not hallucinated."
    ),
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_agent() -> ResearchAgent:
    """Build the agent once per process (loads and indexes the corpus)."""
    return get_default_agent()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Research question")
    top_k: int = Field(4, ge=1, le=10, description="Chunks to retrieve")


class CitationOut(BaseModel):
    chunk_id: str
    doc_id: str
    company: str
    doc_type: str
    date: str
    source: str
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut]
    confidence: str
    mode: str
    refused: bool
    retrieved_chunks: list[CitationOut]


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    document_ids: list[str]
    backend: str


def _to_out(result: AskResult) -> AskResponse:
    return AskResponse(
        question=result.question,
        answer=result.answer,
        citations=[CitationOut(**c.__dict__) for c in result.citations],
        confidence=result.confidence,
        mode=result.mode,
        refused=result.refused,
        retrieved_chunks=[CitationOut(**c.__dict__) for c in result.retrieved_chunks],
    )


@app.get("/health")
def health() -> dict:
    """Liveness check. Returns corpus size and backend info when loaded."""
    agent = get_agent()
    return {
        "status": "ok",
        "documents": len(agent.documents),
        "chunks": len(agent.chunks),
        "backend": agent.backend,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer *request.question* from the indexed documents."""
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="question must not be blank")
    agent = get_agent()
    return _to_out(agent.ask(request.question, top_k=request.top_k))


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    """Re-ingest the bundled sample documents and rebuild the index."""
    get_agent.cache_clear()
    agent = get_agent()
    return IngestResponse(
        documents=len(agent.documents),
        chunks=len(agent.chunks),
        document_ids=[d.doc_id for d in agent.documents],
        backend=agent.backend,
    )
