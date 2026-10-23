import math

import pytest

from agentrag.bench.metrics import (
    context_precision_at_k,
    hit_at_k,
    mean_metrics,
    mrr_at_k,
    ndcg_at_k,
    per_query_metrics,
    recall_at_k,
    view_metrics,
)

REL = {"a": 1, "b": 1, "c": 1}


def test_recall_counts_relevant_found_over_all_relevant():
    assert recall_at_k(["a", "x", "b", "y"], REL, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(["a", "x", "b", "y"], REL, k=1) == pytest.approx(1 / 3)


def test_recall_ignores_zero_labels():
    assert recall_at_k(["a"], {"a": 1, "z": 0}, k=5) == 1.0


def test_mrr_uses_first_relevant_within_cutoff():
    assert mrr_at_k(["x", "y", "b"], REL) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "y", "z"], REL) == 0.0
    assert mrr_at_k(["x"] * 10 + ["a"], REL, k=10) == 0.0


def test_ndcg_perfect_and_hand_computed():
    assert ndcg_at_k(["a", "b", "c"], REL) == pytest.approx(1.0)
    # hits at ranks 1 and 3; ideal puts three hits at ranks 1-3
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(["a", "x", "b"], REL) == pytest.approx(dcg / idcg)


def test_ndcg_graded_relevance_rewards_better_ordering():
    graded = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b"], graded) > ndcg_at_k(["b", "a"], graded)


def test_per_query_metrics_scores_missing_queries_as_zero():
    result = per_query_metrics({"q1": ["a"]}, {"q1": {"a": 1}, "q2": {"b": 1}})
    assert result["q1"]["ndcg@10"] == 1.0
    assert result["q2"]["ndcg@10"] == 0.0
    assert mean_metrics(result)["ndcg@10"] == 0.5


def test_hit_at_k_is_binary_and_respects_the_cutoff():
    assert hit_at_k(["x", "a"], REL, k=1) == 0.0
    assert hit_at_k(["x", "a"], REL, k=2) == 1.0
    assert hit_at_k([], REL, k=5) == 0.0


def test_context_precision_matches_hand_computation():
    # relevant at ranks 1 and 3 of 4: (1/1 + 2/3) / 2
    assert context_precision_at_k(["a", "x", "b", "y"], REL, k=4) == pytest.approx((1 + 2 / 3) / 2)
    # a single relevant doc at rank 3 has precision@3 = 1/3
    assert context_precision_at_k(["x", "y", "a", "z"], REL, k=4) == pytest.approx(1 / 3)
    assert context_precision_at_k(["x", "y"], REL, k=4) == 0.0
    # the window cuts off relevant documents beyond rank k
    assert context_precision_at_k(["x", "y", "z", "w", "a"], REL, k=4) == 0.0


def test_context_precision_rewards_earlier_relevant_context():
    assert context_precision_at_k(["a", "x", "y", "z"], REL) > context_precision_at_k(["x", "y", "z", "a"], REL)


def test_view_metrics_reports_every_column_and_recall_equals_context_recall():
    row = view_metrics(["a", "x", "b", "y", "c"], REL)
    assert {"hit@1", "hit@5", "hit@10", "context_precision@4", "context_recall@4", "ndcg@10"} <= row.keys()
    assert row["context_recall@4"] == pytest.approx(recall_at_k(["a", "x", "b", "y", "c"], REL, 4))
    assert row["hit@1"] == 1.0
