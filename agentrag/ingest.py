"""Document loading and chunking (the 'knowledge base' side of the RAG pipeline)."""
from __future__ import annotations

import io
import os
import re
import uuid
from collections.abc import Callable, Sequence
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


def _pdf_text(source) -> str:
    """Text of a PDF given a path or a binary file object (shared by disk and in-memory ingest)."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pypdf is required to ingest PDF files") from exc

    reader = PdfReader(source)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_pdf(path: Path) -> str:
    return _pdf_text(str(path))


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


# -- in-memory ingest (uploaded files; nothing touches the disk) ---------------------------------


class UnreadableDocument(ValueError):
    """A file could not be turned into text. ``reason`` is written to be shown to a user."""

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"{name}: {reason}")
        self.name, self.reason = name, reason


def extract_text(filename: str, data: bytes) -> str:
    """Text of an uploaded file given its name and raw bytes (PDF, Markdown or plain text).

    Raises ``UnreadableDocument`` with a user-presentable reason for an unsupported type, an empty file, a corrupt
    or encrypted PDF, a PDF with no text layer, or a binary file with a text extension.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnreadableDocument(filename, f"unsupported file type '{suffix or 'none'}' (use PDF, MD or TXT)")
    if not data:
        raise UnreadableDocument(filename, "the file is empty")

    if suffix == ".pdf":
        try:
            text = _pdf_text(io.BytesIO(data))
        except RuntimeError:
            raise  # pypdf itself is missing: an installation problem, not a bad file
        except Exception as exc:  # noqa: BLE001 - pypdf raises many types for corrupt / encrypted files
            raise UnreadableDocument(
                filename, f"could not read this PDF ({type(exc).__name__}); it may be corrupt or password-protected"
            ) from exc
        empty_reason = "the PDF has no extractable text (a scanned document needs OCR first)"
    else:
        text = data.decode("utf-8", errors="ignore")
        if "\x00" in text[:2000]:
            raise UnreadableDocument(filename, "this looks like a binary file, not text")
        empty_reason = "the file contains no text"

    text = text.strip()
    if not text:
        raise UnreadableDocument(filename, empty_reason)
    return text


@dataclass
class FileReport:
    """What happened to one uploaded file."""

    name: str
    status: str  # "indexed" | "partial" | "skipped" | "unreadable"
    chunks_available: int | None  # None when the file was not read at all
    chunks_indexed: int
    detail: str = ""


@dataclass
class IngestResult:
    chunks: List[Chunk]
    files: List[FileReport]
    cap: int | None
    truncated: bool

    @property
    def indexed(self) -> int:
        return len(self.chunks)

    def summary(self) -> str:
        """An exact, user-presentable account of what was and was not indexed."""
        unreadable = [f for f in self.files if f.status == "unreadable"]
        skipped = [f for f in self.files if f.status == "skipped"]
        partial = [f for f in self.files if f.status == "partial"]
        parts = [f"Indexed {self.indexed:,} chunks from {sum(f.chunks_indexed > 0 for f in self.files)} file(s)."]
        if self.truncated:
            parts.append(f"The cap of {self.cap:,} chunks was reached, so the index is incomplete.")
        for f in partial:
            parts.append(
                f"{f.name}: indexed {f.chunks_indexed:,} of {f.chunks_available:,} chunks (the rest was left out)."
            )
        if skipped:
            parts.append("Not read because the cap was already reached: " + ", ".join(f.name for f in skipped) + ".")
        for f in unreadable:
            parts.append(f"{f.name}: skipped, {f.detail}.")
        return " ".join(parts)


def build_corpus_from_files(
    files: Sequence[tuple[str, bytes]],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    max_chunks: int | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> IngestResult:
    """Chunk uploaded files (name, bytes) in order, stopping at ``max_chunks``.

    Reuses the same sentence-aware chunker and PDF reader as disk ingest. Once the cap is reached the remaining
    files are not read; a file that straddles the cap is indexed up to it. Nothing is dropped silently: every file
    gets a ``FileReport``, and ``IngestResult.summary()`` states exactly what was indexed. Duplicate file names are
    disambiguated so citations stay unambiguous. ``progress(fraction, message)`` is called as files are processed.
    """
    chunks: List[Chunk] = []
    reports: List[FileReport] = []
    truncated, seen = False, {}

    for index, (raw_name, data) in enumerate(files):
        seen[raw_name] = seen.get(raw_name, 0) + 1
        name = raw_name if seen[raw_name] == 1 else f"{raw_name} ({seen[raw_name]})"
        if progress:
            progress(index / max(1, len(files)), f"Reading {name}")

        if max_chunks is not None and len(chunks) >= max_chunks:
            truncated = True
            reports.append(FileReport(name, "skipped", None, 0, "not read: the chunk cap was already reached"))
            continue
        try:
            text = extract_text(raw_name, data)
        except UnreadableDocument as exc:
            reports.append(FileReport(name, "unreadable", None, 0, exc.reason))
            continue

        pieces = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        room = len(pieces) if max_chunks is None else max_chunks - len(chunks)
        taken = pieces[:room]
        for position, piece in enumerate(taken):
            chunks.append(Chunk(id=str(uuid.uuid4()), source=name, text=piece, position=position))
        status = "indexed" if len(taken) == len(pieces) else "partial"
        truncated = truncated or status == "partial"
        reports.append(FileReport(name, status, len(pieces), len(taken)))

    if progress:
        progress(1.0, "Chunking complete")
    return IngestResult(chunks=chunks, files=reports, cap=max_chunks, truncated=truncated)
