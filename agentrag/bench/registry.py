"""The named retrieval systems the benchmark scripts can run.

Each factory receives ``get``, which returns the shared instance of another
system, so a component (e.g. BM25) is built and indexed once however many
composite systems use it.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from agentrag.bench.systems import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRRFRetriever,
    Retriever,
    StoreRetriever,
    TfidfRetriever,
)
from agentrag.vectorstore import BGE_QUERY_PREFIX, Bm25VectorStore, HybridVectorStore

CACHE = Path(".bench_cache")
RESULTS = Path("docs/results")

FACTORIES: dict[str, Callable[[Callable[[str], Retriever]], Retriever]] = {
    "tfidf": lambda get: TfidfRetriever(),
    "bm25": lambda get: BM25Retriever(),
    "minilm": lambda get: DenseRetriever("sentence-transformers/all-MiniLM-L6-v2", cache_dir=CACHE / "emb"),
    "bge-small": lambda get: DenseRetriever(
        "BAAI/bge-small-en-v1.5", query_prefix=BGE_QUERY_PREFIX, cache_dir=CACHE / "emb"
    ),
    "hybrid-minilm": lambda get: HybridRRFRetriever([get("bm25"), get("minilm")]),
    "hybrid-bge": lambda get: HybridRRFRetriever([get("bm25"), get("bge-small")]),
    "hybrid-bge-rerank": lambda get: CrossEncoderReranker(get("hybrid-bge")),
    # The app's shipped backends run through the same harness, to confirm the product code
    # scores like the benchmarked components (no embedding cache here, so app-hybrid re-encodes).
    "app-bm25": lambda get: StoreRetriever("app-bm25", Bm25VectorStore),
    "app-hybrid": lambda get: StoreRetriever("app-hybrid", HybridVectorStore),
    # Replication of the BEIR paper's "BM25+CE" setup (BM25 top-100, 6-layer MiniLM MS MARCO cross-encoder).
    # A validation of the reranker implementation, not part of the main ablation family.
    "bm25-rerank100": lambda get: CrossEncoderReranker(get("bm25"), rerank_depth=100),
}


def make_getter() -> Callable[[str], Retriever]:
    """Return ``get(key)`` that builds each system once and then shares it."""
    instances: dict[str, Retriever] = {}

    def get(key: str) -> Retriever:
        if key not in instances:
            instances[key] = FACTORIES[key](get)
        return instances[key]

    return get


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]
