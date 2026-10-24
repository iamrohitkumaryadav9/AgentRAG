"""The logic behind the Streamlit demo (app.py), kept free of Streamlit so it can be tested directly.

Design rules for a hosted app that handles other people's documents and API keys:

* **Keys.** A visitor's Gemini key is passed straight to the Gemini client and nowhere else. This module never
  reads or writes ``os.environ``, never logs, never caches and never writes a file; ``make_llm`` builds a client
  that has no response cache, no spend ledger and no budget file. Error text shown to a visitor is scrubbed of the
  key and of anything shaped like a Google key before it is displayed.
* **Caps.** The index is bounded by *chunks*, not file count, because chunks are what cost memory and (for the
  hybrid retriever) embedding time. Hitting a cap is never silent: ``IngestResult.summary()`` says exactly what was
  and was not indexed.
* **No hidden global state.** Everything a session needs is in the objects returned here, which the app keeps in
  ``st.session_state``; metrics are request-scoped (see ``observability``), so concurrent sessions do not mix.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .config import settings
from .graph import run_query
from .ingest import Chunk, IngestResult, build_corpus_from_files
from .llm import LLM, ExtractiveLLM, GeminiLLM
from .vectorstore import (
    BGE_QUERY_PREFIX,
    HYBRID_DENSE_MODEL,
    Bm25VectorStore,
    DenseVectorStore,
    HybridVectorStore,
    VectorStore,
)

MAX_QUESTION_CHARS = 1000
BACKENDS = ("bm25", "hybrid")
DEFAULT_CHUNK_CAPS = {"bm25": 5000, "hybrid": 1000}  # hybrid is lower: every chunk is embedded on CPU


class NoIndexableText(ValueError):
    """None of the provided files produced any text. The message lists why, file by file."""


class QuestionError(ValueError):
    """The question itself is unusable (empty or too long). The message is user-presentable."""


def chunk_cap(backend: str) -> int:
    """Maximum chunks to index for a backend. Override with AGENTRAG_APP_MAX_CHUNKS_BM25 / _HYBRID."""
    default = DEFAULT_CHUNK_CAPS[backend]
    raw = os.getenv(f"AGENTRAG_APP_MAX_CHUNKS_{backend.upper()}", "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else default


def dense_available() -> bool:
    """True when the hybrid retriever's heavy dependencies are importable (without importing them)."""
    return all(importlib.util.find_spec(name) is not None for name in ("sentence_transformers", "faiss"))


# -- indexing -------------------------------------------------------------------------------------


@dataclass
class IndexBundle:
    store: VectorStore
    ingest: IngestResult
    backend: str
    signature: str
    seconds: float

    @property
    def chunks(self) -> list[Chunk]:
        return self.ingest.chunks


def documents_signature(files: Sequence[tuple[str, bytes]], backend: str) -> str:
    """Fingerprint of (documents, backend) so the index is rebuilt only when one of them changes."""
    digest = hashlib.sha256(backend.encode())
    for name, data in files:
        digest.update(name.encode("utf-8", errors="replace"))
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()[:16]


def build_index(
    files: Sequence[tuple[str, bytes]],
    backend: str,
    progress: Callable[[float, str], None] | None = None,
    dense_model=None,
) -> IndexBundle:
    """Chunk the files (up to the backend's cap) and build the retrieval index. Raises ``NoIndexableText``.

    ``progress(fraction, message)`` drives a progress bar: reading and chunking files is the first 80%, building the
    index the rest. ``dense_model`` is an already-loaded SentenceTransformer shared between sessions.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}")
    started = time.perf_counter()

    def scaled(fraction: float, message: str) -> None:
        if progress:
            progress(0.8 * fraction, message)

    result = build_corpus_from_files(
        files,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_chunks=chunk_cap(backend),
        progress=scaled,
    )
    if not result.chunks:
        raise NoIndexableText(result.summary() or "None of the files contained any text.")

    if progress:
        progress(0.85, "Building the retrieval index" + (" (embedding chunks)" if backend == "hybrid" else ""))
    store: VectorStore
    if backend == "hybrid":
        dense = DenseVectorStore(model_name=HYBRID_DENSE_MODEL, query_prefix=BGE_QUERY_PREFIX, model=dense_model)
        store = HybridVectorStore(parts=[Bm25VectorStore(), dense])
    else:
        store = Bm25VectorStore()
    store.build(result.chunks)
    if progress:
        progress(1.0, "Index ready")
    return IndexBundle(store, result, backend, documents_signature(files, backend), time.perf_counter() - started)


# -- generation backend ------------------------------------------------------------------------------


def make_llm(api_key: str | None) -> LLM:
    """The visitor's Gemini if they gave a key, otherwise the offline extractive generator.

    The key goes straight into the client. It is not placed in the environment, and the environment is never
    consulted as a fallback (an empty key means offline mode, not "use the server's key").
    """
    key = (api_key or "").strip()
    if not key:
        return ExtractiveLLM()
    # two retries and a 60 s timeout: a visitor should not wait a minute-plus on a struggling API
    return GeminiLLM(api_key=key, max_retries=2, timeout_seconds=60.0)


# -- answering ------------------------------------------------------------------------------------------


@dataclass
class CitationView:
    source: str
    position: int
    snippet: str
    found: bool  # False if the cited chunk was not among the retrieved ones (a fabricated citation)


@dataclass
class SpanView:
    agent: str
    ms: float
    details: dict = field(default_factory=dict)


@dataclass
class QueryReport:
    question: str
    answer: str
    path: str  # "direct" | "refuse" | "answer"
    confidence: float
    citations: list[CitationView]
    sources: list[tuple[str, float]]
    critic_passed: bool | None  # None when no critic ran (direct / refuse)
    grounding_score: float | None
    critic_reason: str
    pii_flags: list[str]
    bad_citations: list[str]
    attempts: int
    trace: list[str]
    spans: list[SpanView]
    total_ms: float
    est_tokens: int
    retrieval: str
    generation: str


def router_path(state: dict) -> str:
    """Which way the router sent the query: small talk ("direct"), refusal, or a normal answer."""
    if state.get("route") == "direct":
        return "direct"
    if any(step.startswith("refuse") for step in state.get("trace", [])):
        return "refuse"
    return "answer"


def validate_question(question: str) -> str:
    cleaned = (question or "").strip()
    if not cleaned:
        raise QuestionError("Please type a question first.")
    if len(cleaned) > MAX_QUESTION_CHARS:
        raise QuestionError(f"Questions are limited to {MAX_QUESTION_CHARS:,} characters; yours has {len(cleaned):,}.")
    return cleaned


def answer_question(bundle: IndexBundle, question: str, llm: LLM, snippet_chars: int = 400) -> QueryReport:
    """Run one question through the agent graph and package everything the UI shows."""
    cleaned = validate_question(question)
    state = run_query(bundle.store, cleaned, llm=llm)

    payload = state["payload"]
    retrieved = state.get("retrieved", [])
    by_ref = {(r.chunk.source, r.chunk.position): r.chunk for r in retrieved}
    citations = []
    for cite in payload.citations:
        chunk = by_ref.get((cite.source, cite.position))
        text = chunk.text if chunk is not None else ""
        snippet = text[:snippet_chars] + ("…" if len(text) > snippet_chars else "")
        citations.append(CitationView(cite.source, cite.position, snippet, found=chunk is not None))

    guardrail = state.get("guardrail")
    metrics = state.get("metrics")
    spans = [
        SpanView(s.name, s.duration_ms, {k.removeprefix("agentrag."): v for k, v in s.attributes.items()})
        for s in (metrics.spans if metrics else [])
    ]
    return QueryReport(
        question=cleaned,
        answer=state.get("answer", payload.answer),
        path=router_path(state),
        confidence=payload.confidence,
        citations=citations,
        sources=[(r.chunk.source, r.score) for r in retrieved],
        critic_passed=guardrail.passed if guardrail else None,
        grounding_score=guardrail.grounding_score if guardrail else None,
        critic_reason=guardrail.reason if guardrail else "",
        pii_flags=list(state.get("pii_flags", [])),
        bad_citations=list(state.get("bad_citations", [])),
        attempts=state.get("attempts", 0),
        trace=list(state.get("trace", [])),
        spans=spans,
        total_ms=metrics.total_ms if metrics else 0.0,
        est_tokens=metrics.est_tokens if metrics else 0,
        retrieval=bundle.store.backend_name,
        generation=llm.backend_name,
    )


# -- friendly errors ---------------------------------------------------------------------------------------

_KEY_SHAPES = (re.compile(r"AIza[0-9A-Za-z_-]{20,}"), re.compile(r"AQ\.[A-Za-z0-9_-]{20,}"))


def scrub(text: str, secret: str | None = None) -> str:
    """Remove the visitor's key, and anything shaped like a Google API key, from text before it is displayed."""
    out = text
    if secret and len(secret.strip()) >= 6:
        out = out.replace(secret.strip(), "[key hidden]")
    for pattern in _KEY_SHAPES:
        out = pattern.sub("[key hidden]", out)
    return out


@dataclass
class UserError:
    kind: str  # invalid_key | quota | unavailable | timeout | blocked | unknown
    message: str
    detail: str = ""


def friendly_error(exc: BaseException, secret: str | None = None) -> UserError:
    """Turn an API/pipeline exception into a message a visitor can act on. Never echoes the key."""
    from .verify.gemini import RateLimitError, TransientError

    name = type(exc).__name__
    text = str(exc)
    lowered = text.lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    detail = scrub(f"{name}: {text}"[:400], secret)

    if isinstance(exc, RateLimitError) or code == 429 or "resource_exhausted" in lowered or "quota" in lowered:
        return UserError(
            "quota",
            "Gemini says this key has reached its rate or daily quota (HTTP 429). Wait a minute and try again, "
            "check the key's quota in Google AI Studio, or clear the key to continue in offline mode.",
            detail,
        )
    if isinstance(exc, TransientError) or (isinstance(code, int) and 500 <= code < 600) or "unavailable" in lowered:
        return UserError(
            "unavailable",
            "Gemini is temporarily unavailable (a 503 or similar), even after retrying. Try again in a moment, or "
            "clear the key to continue in offline mode.",
            detail,
        )
    if (
        code in (401, 403)
        or "api key not valid" in lowered
        or "api_key_invalid" in lowered
        or "invalid api key" in lowered
        or "no gemini api key" in lowered
        or (code == 400 and "api key" in lowered)
    ):
        return UserError(
            "invalid_key",
            "Gemini rejected this API key. Check that it was copied completely and that the Gemini API is enabled "
            "for its project, or clear the key to use offline mode.",
            detail,
        )
    if "timeout" in name.lower() or "timed out" in lowered or isinstance(exc, TimeoutError):
        return UserError(
            "timeout", "The request to Gemini timed out. Try again, or clear the key to use offline mode.", detail
        )
    return UserError(
        "unknown",
        f"Something went wrong while answering ({name}). Your documents are still indexed; try again, or clear the "
        "key to use offline mode.",
        detail,
    )
