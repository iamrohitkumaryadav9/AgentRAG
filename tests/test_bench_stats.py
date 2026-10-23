import numpy as np
import pytest

from agentrag.bench.stats import holm_adjust, paired_bootstrap


def test_identical_systems_have_zero_difference_and_ci_covers_zero():
    scores = np.random.default_rng(1).random(200)
    result = paired_bootstrap(scores, scores)
    assert result["mean_diff"] == 0.0
    assert result["ci_low"] <= 0.0 <= result["ci_high"]
    assert result["p_value"] == 1.0


def test_consistent_improvement_is_significant_and_ci_excludes_zero():
    base = np.random.default_rng(2).random(300)
    result = paired_bootstrap(base + 0.05, base)
    assert result["mean_diff"] == pytest.approx(0.05)
    assert result["ci_low"] > 0
    assert result["p_value"] < 0.001


def test_noisy_difference_with_zero_mean_is_not_significant():
    rng = np.random.default_rng(3)
    b = rng.random(300)
    noise = rng.normal(0, 0.2, 300)
    a = b + (noise - noise.mean())  # per-query differences are noisy but sum to exactly zero
    result = paired_bootstrap(a, b)
    assert result["mean_diff"] == pytest.approx(0.0, abs=1e-12)
    assert result["ci_low"] < 0 < result["ci_high"]
    assert result["p_value"] > 0.5


def test_ci_brackets_observed_mean_and_is_seed_deterministic():
    a, b = np.random.default_rng(4).random(100), np.random.default_rng(5).random(100)
    first, second = paired_bootstrap(a, b, seed=7), paired_bootstrap(a, b, seed=7)
    assert first == second
    assert first["ci_low"] <= first["mean_diff"] <= first["ci_high"]


def test_bootstrap_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])


def test_holm_matches_hand_computation_and_never_decreases_p():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    # sorted: 0.01*3=0.03, 0.03*2=0.06, 0.04*1 -> max(0.06, 0.04)=0.06
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert all(adj >= p for adj, p in zip(adjusted, [0.01, 0.04, 0.03], strict=True))
