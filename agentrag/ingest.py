"""Document loading and chunking (the 'knowledge base' side of the RAG pipeline)."""
from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Chunk:
    id: str
    source: str
    text: str
    position: int
    metadata: dict = field(default_factory=dict)


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pypdf is required to ingest PDF files") from exc

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(directory: str | os.PathLike) -> List[dict]:
    """Load every supported file under `directory` into {source, text} records."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Docs directory not found: {directory}")

    documents = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.suffix.lower() == ".pdf":
            text = _read_pdf(path)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
        text = text.strip()
        if text:
            documents.append({"source": str(path.relative_to(directory)), "text": text})
    return documents


def _split_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter -- avoids pulling in a heavy NLP dependency
    # just to break text at '.', '?', '!' followed by whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.?!])\s+", text)
    return [s for s in sentences if s]


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    """Sentence-aware sliding-window chunking with overlap.

    Splitting on sentence boundaries (rather than raw character windows)
    keeps each chunk semantically coherent, which matters for retrieval
    quality -- a half-sentence chunk is a poor unit to embed or cite.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sentences = _split_sentences(text)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            # start next chunk with overlap tail of the previous chunk
            overlap_tail = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{overlap_tail} {sentence}".strip()
        else:
            # single sentence longer than chunk_size: hard-split it
            for i in range(0, len(sentence), chunk_size - chunk_overlap):
                chunks.append(sentence[i : i + chunk_size])
            current = ""

    if current:
        chunks.append(current)

    return chunks


def build_corpus(directory: str | os.PathLike, chunk_size: int = 800, chunk_overlap: int = 120) -> List[Chunk]:
    """Load documents from `directory` and return them as retrieval-ready Chunks."""
    documents = load_documents(directory)
    corpus: List[Chunk] = []
    for doc in documents:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for position, piece in enumerate(pieces):
            corpus.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    source=doc["source"],
                    text=piece,
                    position=position,
                )
            )
    return corpus
