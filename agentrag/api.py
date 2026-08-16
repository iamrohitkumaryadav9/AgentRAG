"""FastAPI service exposing the RAG pipeline for production-style deployment.

    POST /ingest   {"directory": "data/sample_docs"}  -> (re)builds the index
    POST /query    {"question": "..."}                 -> runs the agent graph
    GET  /health                                        -> liveness probe
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import settings
from .evaluation import evaluate
from .graph import run_query
from .ingest import build_corpus
from .llm import get_llm
from .vectorstore import VectorStore, get_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AgentRAG", version="0.1.0", description="A small, self-correcting multi-agent RAG assistant.")

_state = {"vector_store": None, "llm": None, "num_chunks": 0}


class IngestRequest(BaseModel):
    directory: str = settings.docs_dir


class IngestResponse(BaseModel):
    num_chunks: int
    backend: str


class QueryRequest(BaseModel):
    question: str


class SourceRef(BaseModel):
    source: str
    score: float


class CitationRef(BaseModel):
    source: str
    position: int


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationRef]
    confidence: float
    sources: List[SourceRef]
    grounded: bool
    grounding_score: float
    attempts: int
    pii_flags: List[str]
    bad_citations: List[str]
    trace: List[str]
    telemetry: dict


def _get_vector_store() -> VectorStore:
    if _state["vector_store"] is None:
        raise HTTPException(status_code=400, detail="Index not built yet. Call POST /ingest first.")
    return _state["vector_store"]


@app.get("/health")
def health():
    return {"status": "ok", "index_built": _state["vector_store"] is not None}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    try:
        corpus = build_corpus(req.directory, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        store = get_vector_store(settings.vector_backend)
        store.build(corpus)
        _state["vector_store"] = store
        _state["llm"] = get_llm(settings.llm_backend)
        _state["num_chunks"] = len(corpus)
        logger.info("Indexed %d chunks from %s using %s backend", len(corpus), req.directory, store.backend_name)
        return IngestResponse(num_chunks=len(corpus), backend=store.backend_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    store = _get_vector_store()
    result = run_query(store, req.question, llm=_state["llm"])
    guardrail = result.get("guardrail")
    payload = result.get("payload")
    metrics = result.get("metrics")
    sources = [SourceRef(source=r.chunk.source, score=r.score) for r in result.get("retrieved", [])]
    citations = [CitationRef(source=c.source, position=c.position) for c in (payload.citations if payload else [])]

    return QueryResponse(
        answer=result.get("answer", ""),
        citations=citations,
        confidence=payload.confidence if payload else 0.0,
        sources=sources,
        grounded=guardrail.passed if guardrail else True,
        grounding_score=guardrail.grounding_score if guardrail else 1.0,
        attempts=result.get("attempts", 0),
        pii_flags=result.get("pii_flags", []),
        bad_citations=result.get("bad_citations", []),
        trace=result.get("trace", []),
        telemetry=metrics.as_dict() if metrics else {},
    )


@app.post("/eval")
def run_evaluation():
    """Run the golden-set evaluation and return AI quality metrics.

    Exposing this as an endpoint (not just a CI script) means the deployed
    service can be re-scored against its own knowledge base at any time --
    useful for catching quality drift after the corpus changes.
    """
    store = _get_vector_store()
    report = evaluate(store, llm=_state["llm"])
    return report["metrics"]
