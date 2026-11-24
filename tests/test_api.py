"""Tests for the FastAPI service (TestClient, no server needed)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["documents"] == 3
    assert body["chunks"] >= 6


def test_ask_returns_answer_with_citations():
    response = client.post("/ask", json={"question": "What was Northwind Cloud Systems revenue in Q3 2026?"})
    assert response.status_code == 200
    body = response.json()
    assert "$412.8 million" in body["answer"]
    assert body["citations"]
    assert body["citations"][0]["doc_id"] == "northwind-q3-2026-earnings-call"
    assert body["refused"] is False
    assert body["mode"] == "mock-extractive"
    assert body["confidence"] in {"low", "medium", "high"}


def test_ask_out_of_scope_refuses():
    response = client.post("/ask", json={"question": "What was Tesla's revenue in Q3 2026?"})
    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert "Insufficient evidence" in body["answer"]


def test_ask_rejects_blank_question():
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_ingest_returns_counts():
    response = client.post("/ingest")
    assert response.status_code == 200
    body = response.json()
    assert body["documents"] == 3
    assert body["chunks"] >= 6
    assert "northwind-q3-2026-earnings-call" in body["document_ids"]
