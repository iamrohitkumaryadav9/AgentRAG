import math

import pytest

from agentrag.bench.analysis import (
    classify_by_dense_rank,
    dcg_contribution,
    loss_driver,
    overtakers,
    rank_of,
    worst_losses,
)


def test_rank_of_is_one_based_and_none_when_absent():
    assert rank_of(["a", "b"], "b") == 2
    assert rank_of(["a", "b"], "z") is None


def test_dcg_contribution_is_zero_beyond_cutoff():
    assert dcg_contribution(1) == 1.0
    assert dcg_contribution(3) == pytest.approx(1 / math.log2(4))
    assert dcg_contribution(11) == 0.0
    assert dcg_contribution(None) == 0.0


def test_loss_driver_picks_the_gold_doc_that_lost_most():
    winner = ["g1", "g2"] + ["x"] * 8
    loser = ["x"] * 4 + ["g1", "x", "x", "x", "x", "g2"]  # g1 drops 1 -> 5, g2 drops 2 -> 10
    assert loss_driver(["g1", "g2"], winner, loser) == "g1"


def test_overtakers_are_ahead_in_loser_but_behind_in_winner():
    winner = ["a", "g", "b", "c"]  # gold at 2; 'a' already ahead of it
    loser = ["a", "b", "x", "g"]   # 'a' ahead (not an overtaker), 'b' and 'x' overtook gold
    assert overtakers("g", winner, loser) == ["b", "x"]
    assert overtakers("missing", winner, loser) == []


@pytest.mark.parametrize(
    ("dense_rank", "group"),
    [(None, "dense-blind-spot"), (51, "dense-blind-spot"), (50, "dense-weak"), (11, "dense-weak"),
     (10, "consensus-demotion"), (1, "consensus-demotion")],
)
def test_group_boundaries(dense_rank, group):
    assert classify_by_dense_rank(dense_rank) == group


def test_worst_losses_orders_by_size_and_reports_totals():
    winner = {"q1": 1.0, "q2": 0.5, "q3": 0.4, "q4": 0.2}
    loser = {"q1": 0.4, "q2": 0.5, "q3": 0.1, "q4": 0.9}
    losses, totals = worst_losses(winner, loser, n=1)
    assert losses == [("q1", pytest.approx(-0.6))]
    assert totals["queries_lost"] == 2
    assert totals["queries_won"] == 1
    assert totals["queries_tied"] == 1
    assert totals["total_gain"] == pytest.approx(0.7)
