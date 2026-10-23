"""Rank fusion shared by the app's hybrid retriever and the benchmark harness."""
from __future__ import annotations


def rrf_fuse(rankings: list[list[str]], k: int = 60, limit: int = 100) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank), rank from 1.

    Rank-based, so it needs no score normalisation across systems whose raw
    scores (BM25, cosine) live on incompatible scales. Ties break by first
    appearance, keeping the result deterministic.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda item: -item[1])  # sort is stable
    return ordered[:limit]
