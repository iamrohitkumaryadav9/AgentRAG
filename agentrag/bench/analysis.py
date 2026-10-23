"""Rule-based error analysis: why does one system lose to another on a query?

The grouping is mechanical so it can be reproduced and audited rather than
labelled by hand after looking at the answers:

1. *Loss driver*: the gold document whose nDCG@10 contribution dropped most
   between the winning system (BM25) and the losing one (hybrid).
2. *Group*: where the dense retriever ranked that driver document.

    dense-blind-spot    dense rank > 50, or not in its top 100
    dense-weak          dense rank 11-50
    consensus-demotion  dense rank <= 10: the dense retriever also found it,
                        but other documents both retrievers like outranked it
                        once their lists were fused
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

GROUPS = ("dense-blind-spot", "dense-weak", "consensus-demotion")


def rank_of(ranking: Sequence[str], doc_id: str) -> int | None:
    """1-based rank of ``doc_id``, or None if it is not in the ranking."""
    try:
        return ranking.index(doc_id) + 1
    except ValueError:
        return None


def dcg_contribution(rank: int | None, k: int = 10) -> float:
    """Discounted gain of one relevant document at ``rank`` (0 beyond the cutoff)."""
    return 1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0


def loss_driver(
    gold: Sequence[str], winner_ranking: Sequence[str], loser_ranking: Sequence[str]
) -> str:
    """Gold document that lost the most nDCG contribution going winner -> loser."""
    return max(
        gold,
        key=lambda doc: dcg_contribution(rank_of(winner_ranking, doc)) - dcg_contribution(rank_of(loser_ranking, doc)),
    )


def overtakers(driver: str, winner_ranking: Sequence[str], loser_ranking: Sequence[str]) -> list[str]:
    """Documents ranked ahead of ``driver`` by the loser but behind it by the winner.

    These are the documents that actually displaced the gold document when
    the two systems' behaviour is compared, in the loser's order.
    """
    loser_pos, winner_pos = rank_of(loser_ranking, driver), rank_of(winner_ranking, driver)
    if loser_pos is None:
        return []
    ahead = loser_ranking[: loser_pos - 1]
    winner_index = {doc: i + 1 for i, doc in enumerate(winner_ranking)}
    limit = winner_pos if winner_pos is not None else len(winner_ranking) + 1
    return [doc for doc in ahead if winner_index.get(doc, len(winner_ranking) + 1) > limit]


def rare_term_queries(
    query_terms: Mapping[str, Sequence[str]], doc_freq: Mapping[str, int], max_df: int
) -> list[str]:
    """Query ids containing at least one term that occurs in 1..``max_df`` documents.

    A reproducible operational definition of "the query hinges on a rare exact
    term" (gene symbols, acronyms, rare names). Terms absent from the corpus
    (df = 0) cannot help any lexical retriever and are ignored.
    """
    return [
        qid for qid, terms in query_terms.items() if any(0 < doc_freq.get(t, 0) <= max_df for t in set(terms))
    ]


def blind_spots(per_query_recall: Mapping[str, float]) -> list[str]:
    """Query ids for which a system retrieved none of the gold documents."""
    return [qid for qid, recall in per_query_recall.items() if recall == 0]


def classify_by_dense_rank(dense_rank: int | None) -> str:
    if dense_rank is None or dense_rank > 50:
        return "dense-blind-spot"
    if dense_rank > 10:
        return "dense-weak"
    return "consensus-demotion"


def worst_losses(
    per_query_winner: Mapping[str, float], per_query_loser: Mapping[str, float], n: int = 20
) -> tuple[list[tuple[str, float]], dict[str, float]]:
    """The ``n`` biggest per-query losses (ties broken by query id) plus totals."""
    diffs = {qid: per_query_loser[qid] - per_query_winner[qid] for qid in per_query_winner}
    losses = sorted(((qid, d) for qid, d in diffs.items() if d < -1e-9), key=lambda item: (item[1], item[0]))
    totals = {
        "queries_lost": len(losses),
        "queries_won": sum(1 for d in diffs.values() if d > 1e-9),
        "queries_tied": sum(1 for d in diffs.values() if abs(d) <= 1e-9),
        "total_loss": sum(d for _, d in losses),
        "total_gain": sum(d for d in diffs.values() if d > 1e-9),
    }
    return losses[:n], totals
