"""Document ingestion and chunking.

Reads plain-text / Markdown files, attaches structured metadata
(company, document type, date, source file), and splits text into
overlapping chunks so a fact is rarely cut in half at a boundary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}


@dataclass
class Document:
    """One source document."""

    doc_id: str
    title: str
    company: str
    doc_type: str  # e.g. "earnings-call", "10-k-excerpt", "press-release"
    date: str  # ISO date, e.g. "2026-10-28"
    source: str  # file name / path
    text: str


@dataclass
class Chunk:
    """One retrievable chunk of a document."""

    chunk_id: str
    doc_id: str
    company: str
    doc_type: str
    date: str
    source: str
    text: str
    chunk_index: int = 0


def parse_metadata(text: str, source: str) -> dict[str, str]:
    """Parse a simple ``Key: value`` header at the top of a sample doc.

    Lines such as ``Company: Northwind Cloud Systems`` are read until the
    first blank line. Missing keys fall back to ``Unknown`` / ``""``.
    """
    meta: dict[str, str] = {"company": "Unknown", "doc_type": "document", "date": "", "title": Path(source).stem}
    for line in text.splitlines():
        if not line.strip():
            break
        match = re.match(r"^(Company|Type|Date|Title):\s*(.+?)\s*$", line.strip())
        if match:
            key, value = match.group(1).lower(), match.group(2)
            if key == "company":
                meta["company"] = value
            elif key == "type":
                meta["doc_type"] = value
            elif key == "date":
                meta["date"] = value
            elif key == "title":
                meta["title"] = value
    return meta


def load_document(path: Path) -> Document:
    """Load one .txt/.md file into a Document.

    The ``Key: value`` header is parsed into metadata and stripped from
    the body, so retrieved chunks contain prose, not the header itself.
    """
    text = path.read_text(encoding="utf-8")
    meta = parse_metadata(text, path.name)
    body = text.split("\n\n", 1)[1].strip() if "\n\n" in text else text
    return Document(
        doc_id=path.stem,
        title=meta["title"],
        company=meta["company"],
        doc_type=meta["doc_type"],
        date=meta["date"],
        source=path.name,
        text=body,
    )


def ingest_directory(directory: str | Path) -> list[Document]:
    """Load every supported file in *directory*, sorted by file name."""
    directory = Path(directory)
    docs: list[Document] = []
    if not directory.exists():
        return docs
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            docs.append(load_document(path))
    return docs


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split *text* into overlapping chunks of ~``chunk_size`` characters.

    Chunking is sentence-aware: sentences are packed into chunks in
    order, and each chunk after the first re-uses the last sentence(s)
    of the previous chunk (up to ``overlap`` characters), so a fact is
    never split mid-sentence at a boundary. A single sentence longer
    than ``chunk_size`` falls back to a word-boundary character window.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def _flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append(" ".join(current))
        current, current_len = [], 0

    def _split_long_sentence(sentence: str) -> list[str]:
        # Word-boundary character window, as a fallback for one huge sentence.
        pieces: list[str] = []
        start = 0
        while start < len(sentence):
            end = min(start + chunk_size, len(sentence))
            if end < len(sentence):
                space = sentence.rfind(" ", start, end)
                if space > start:
                    end = space
            pieces.append(sentence[start:end].strip())
            if end >= len(sentence):
                break
            start = max(end - overlap, start + 1)
        return pieces

    for sentence in sentences:
        if len(sentence) > chunk_size:
            _flush()
            chunks.extend(_split_long_sentence(sentence))
            continue
        if current and current_len + 1 + len(sentence) > chunk_size:
            _flush()
            # Overlap: carry the last sentence(s) of the previous chunk.
            overlap_sentences: list[str] = []
            overlap_len = 0
            prev_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunks[-1]) if s.strip()]
            for prev in reversed(prev_sentences):
                if overlap_len + len(prev) + 1 > overlap and overlap_sentences:
                    break
                overlap_sentences.insert(0, prev)
                overlap_len += len(prev) + 1
                if overlap_len >= overlap:
                    break
            current = list(overlap_sentences)
            current_len = sum(len(s) + 1 for s in current)
        current.append(sentence)
        current_len += len(sentence) + 1
    _flush()
    return [c for c in chunks if c]


def chunk_document(doc: Document, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Chunk one document, giving each chunk a stable ``<doc_id>#<n>`` id."""
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#{i}",
            doc_id=doc.doc_id,
            company=doc.company,
            doc_type=doc.doc_type,
            date=doc.date,
            source=doc.source,
            text=piece,
            chunk_index=i,
        )
        for i, piece in enumerate(chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap))
    ]


def ingest_and_chunk(directory: str | Path, chunk_size: int = 800, overlap: int = 150) -> tuple[list[Document], list[Chunk]]:
    """Ingest a directory and return ``(documents, chunks)``."""
    docs = ingest_directory(directory)
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return docs, chunks
