"""IR metrics, computed per query so they can be bootstrapped later.

Definitions follow BEIR's evaluation protocol:

    recall@k   relevant docs in the top k / all relevant docs for the query
    mrr@10     reciprocal rank of the first relevant doc in the top 10
    ndcg@10    DCG with gain = relevance label and a log2(rank + 1) discount,
               normalised by the ideal ranking of that query's judged docs

``qrels`` maps query id -> {doc id -> graded relevance}; only labels > 0 count
as relevant. ``ranking`` is a list of doc ids, best first.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

Qrels = Mapping[str, Mapping[str, int]]
Run = Mapping[str, Sequence[str]]


def recall_at_k(ranking: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    positives = {d for d, rel in relevant.items() if rel > 0}
    if not positives:
        return 0.0
    return len(positives & set(ranking[:k])) / len(positives)


def mrr_at_k(ranking: Sequence[str], relevant: Mapping[str, int], k: int = 10) -> float:
    for rank, doc_id in enumerate(ranking[:k], start=1):
        if relevant.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: Sequence[str], relevant: Mapping[str, int], k: int = 10) -> float:
    gains = [relevant.get(doc_id, 0) for doc_id in ranking[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains) if g > 0)
    ideal = sorted((rel for rel in relevant.values() if rel > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def per_query_metrics(
    run: Run, qrels: Qrels, recall_ks: Sequence[int] = (5, 50, 100)
) -> dict[str, dict[str, float]]:
    """Metrics for every query that has judgments, keyed by query id."""
    out: dict[str, dict[str, float]] = {}
    for qid, relevant in qrels.items():
        ranking = run.get(qid, [])
        row = {f"recall@{k}": recall_at_k(ranking, relevant, k) for k in recall_ks}
        row["mrr@10"] = mrr_at_k(ranking, relevant, 10)
        row["ndcg@10"] = ndcg_at_k(ranking, relevant, 10)
        out[qid] = row
    return out


def mean_metrics(per_query: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    names = next(iter(per_query.values())).keys()
    n = len(per_query)
    return {name: sum(row[name] for row in per_query.values()) / n for name in names}
