"""Pluggable LLM layer.

Two backends:
1. OpenAI (``OPENAI_API_KEY`` set) -- calls the Chat Completions API
   with a system prompt that forbids inventing numbers and requires
   citations to the retrieved chunks.
2. Mock extractive LLM (default, no key, no network) -- composes an
   answer *only* from sentences in the retrieved chunks. It never
   generates a number that is not already in a retrieved chunk, because
   it only ever copies sentences verbatim. If no chunk is relevant it
   says so explicitly. This is the safe offline default and the one
   used by the demo, tests, and evals.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from rag.ingest import Chunk
from rag.store import SearchResult

SYSTEM_PROMPT = (
    "You are a financial research assistant. Answer ONLY using the retrieved "
    "context below. Every factual claim, and especially every number, must come "
    "from the context and be cited as [n] matching the chunk number. If the "
    "context does not contain the answer, say 'Insufficient evidence in the "
    "provided documents.' Do not invent numbers, dates, or company names."
)

# Numbers that the mock LLM is allowed to emit are exactly the numbers
# appearing in the retrieved chunks -- enforced by construction (see below).
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class LLMAnswer:
    answer: str
    mode: str  # "mock-extractive" or "openai"
    used_chunks: list[str]  # chunk_ids actually cited


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def _question_tokens(question: str) -> set[str]:
    from rag.embeddings import tokenize

    return set(tokenize(question))


class MockExtractiveLLM:
    """Offline answerer: copies the most relevant sentences verbatim.

    Guarantees, by construction:
    * No sentence (and therefore no number) is emitted unless it appears
      in a retrieved chunk.
    * Every emitted sentence is tied to its source chunk id.
    """

    mode = "mock-extractive"

    def generate(self, question: str, results: list[SearchResult]) -> LLMAnswer:
        if not results:
            return LLMAnswer(
                answer="Insufficient evidence in the provided documents.",
                mode=self.mode,
                used_chunks=[],
            )
        q_tokens = _question_tokens(question)
        # Weight question tokens by rarity across the retrieved chunks:
        # a token appearing in only one chunk (e.g. "guidance", "q4") is
        # far more discriminating than one in every chunk ("revenue").
        # This is the same IDF idea as the embedder, computed locally.
        import math as _math

        n_chunks = max(len(results), 1)
        df: dict[str, int] = {}
        for result in results:
            for token in set(_question_tokens(result.chunk.text)):
                df[token] = df.get(token, 0) + 1
        weight = {t: _math.log((1 + n_chunks) / (1 + df.get(t, 0))) + 1.0 for t in q_tokens}
        scored: list[tuple[float, str, str]] = []  # (score, sentence, chunk_id)
        for result in results:
            for sentence in _split_sentences(result.chunk.text):
                tokens = set(_question_tokens(sentence))
                overlap = sum(weight.get(t, 0.0) for t in (q_tokens & tokens))
                # Small bonus for sentences containing a number: financial
                # questions are usually answered by a numeric sentence.
                if re.search(r"\d", sentence):
                    overlap += 0.5
                scored.append((overlap, sentence, result.chunk.chunk_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        # Keep only sentences that share at least one content token with
        # the question, or a number-bearing sentence from the top chunk.
        picked: list[tuple[str, str]] = []
        seen: set[str] = set()
        for overlap, sentence, chunk_id in scored:
            if overlap <= 0:
                continue
            if sentence in seen:
                continue
            seen.add(sentence)
            picked.append((sentence, chunk_id))
            if len(picked) >= 3:
                break
        if not picked:
            return LLMAnswer(
                answer="Insufficient evidence in the provided documents.",
                mode=self.mode,
                used_chunks=[],
            )
        # Map chunk_id -> citation number, in first-use order.
        citation_of: dict[str, int] = {}
        lines: list[str] = []
        for sentence, chunk_id in picked:
            if chunk_id not in citation_of:
                citation_of[chunk_id] = len(citation_of) + 1
            lines.append(f"- {sentence} [{citation_of[chunk_id]}]")
        answer = (
            "Based on the retrieved documents:\n" + "\n".join(lines)
            + "\n\nNote: this answer was composed in offline mock mode, "
            "using only sentences from the retrieved chunks above."
        )
        return LLMAnswer(answer=answer, mode=self.mode, used_chunks=list(citation_of.keys()))


class OpenAILLM:
    """OpenAI backend. Requires ``OPENAI_API_KEY`` (never commit it)."""

    mode = "openai"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, question: str, results: list[SearchResult]) -> LLMAnswer:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("openai package is not installed") from exc
        context = "\n\n".join(
            f"[{i + 1}] {r.chunk.company} | {r.chunk.doc_type} | {r.chunk.date} "
            f"| chunk {r.chunk.chunk_id}:\n{r.chunk.text}"
            for i, r in enumerate(results)
        )
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0,
        )
        text = response.choices[0].message.content or ""
        used = [r.chunk.chunk_id for r in results if f"[{results.index(r) + 1}]" in text]
        return LLMAnswer(answer=text, mode=self.mode, used_chunks=used or [r.chunk.chunk_id for r in results])


def get_llm():
    """Return OpenAILLM when a key is configured, else MockExtractiveLLM."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAILLM()
        except Exception:
            pass  # fall back to the safe mock backend
    return MockExtractiveLLM()
