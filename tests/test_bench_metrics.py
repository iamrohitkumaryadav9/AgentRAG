import math

import pytest

from agentrag.bench.metrics import mean_metrics, mrr_at_k, ndcg_at_k, per_query_metrics, recall_at_k

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
