import numpy as np
import pytest

from agentrag.bench.stats import paired_bootstrap_metric, wilson_interval
from agentrag.verify.metrics import (
    accuracy,
    checker_validation,
    cohens_kappa,
    confusion,
    contradiction_recall,
    evidence_prf,
    faithfulness,
    macro_f1,
    per_class,
    retry_stats,
)

S, C, N = "SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"
TRUE = [S, S, C, C, N, N]
PRED = [S, C, C, C, N, S]


def test_confusion_and_per_class_match_hand_computation():
    assert confusion(TRUE, PRED)[S] == {S: 1, C: 1, N: 0}
    pc = per_class(TRUE, PRED)
    assert pc[S]["precision"] == pytest.approx(0.5) and pc[S]["recall"] == pytest.approx(0.5)
    assert pc[C]["precision"] == pytest.approx(2 / 3) and pc[C]["recall"] == 1.0
    assert pc[N]["precision"] == 1.0 and pc[N]["recall"] == pytest.approx(0.5)


def test_macro_f1_accuracy_and_kappa_match_hand_computation():
    assert macro_f1(TRUE, PRED) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert accuracy(TRUE, PRED) == pytest.approx(4 / 6)
    assert cohens_kappa(TRUE, PRED) == pytest.approx(0.5)  # po = 2/3, pe = 1/3
    assert contradiction_recall(TRUE, PRED) == 1.0


def test_perfect_and_chance_level_agreement():
    assert macro_f1(TRUE, TRUE) == 1.0 and cohens_kappa(TRUE, TRUE) == 1.0
    always_support = [S] * 6
    assert cohens_kappa(TRUE, always_support) == pytest.approx(0.0)
    assert contradiction_recall(TRUE, always_support) == 0.0


def test_a_class_with_no_predictions_scores_zero_f1_not_an_error():
    assert per_class([S, C, N], [S, S, S])[C]["f1"] == 0.0


def test_retry_stats_separates_fixes_from_harm():
    gold = [S, S, C, C, S, N]
    initial = [S, C, C, S, C, N]  # wrong at index 1, 3, 4
    flagged = [False, True, False, True, True, True]  # idx 5 flagged although it was right
    final = [S, S, C, C, C, S]  # 1 and 3 fixed, 4 still wrong, 5 harmed
    stats = retry_stats(gold, initial, flagged, final)
    assert stats["flagged"] == 4 and stats["critic_precision"] == pytest.approx(3 / 4)
    assert stats["critic_recall"] == pytest.approx(3 / 3)
    assert stats["fixed"] == 2 and stats["fix_rate"] == pytest.approx(2 / 3)
    assert stats["harmed"] == 1 and stats["harm_rate"] == pytest.approx(1 / 1)
    assert stats["net_fixed_per_100_claims"] == pytest.approx(100 * (2 - 1) / 6)


def test_retry_stats_handles_a_critic_that_never_flags():
    stats = retry_stats([S, C], [S, S], [False, False], [S, S])
    assert stats["flagged"] == 0 and stats["critic_precision"] is None and stats["fix_rate"] is None


def test_faithfulness_excludes_undefined_entries_from_the_denominator():
    assert faithfulness([True, True, False, None, None]) == {"faithful": 2, "judged": 3, "rate": pytest.approx(2 / 3)}
    assert faithfulness([None, None])["rate"] is None


def test_evidence_prf_is_micro_averaged_and_penalises_uncited_gold():
    pairs = [({("1", 0), ("1", 1)}, {("1", 0)}), (set(), {("2", 3), ("2", 4)})]
    result = evidence_prf(pairs)
    assert result["precision"] == pytest.approx(1 / 2) and result["recall"] == pytest.approx(1 / 3)
    assert result["f1"] == pytest.approx(2 * 0.5 * (1 / 3) / (0.5 + 1 / 3))


def test_checker_validation_reports_per_class_recall():
    expected = ["entails", "entails", "contradicts", "neutral"]
    result = checker_validation(expected, ["entails", "neutral", "contradicts", "neutral"])
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["per_class"]["entails"]["recall"] == pytest.approx(0.5)


def test_paired_bootstrap_on_macro_f1_covers_zero_for_identical_systems_and_excludes_it_for_a_real_gap():
    rng = np.random.default_rng(0)
    truth = list(rng.choice([S, C, N], size=200))
    good = list(truth)
    bad = [t if rng.random() < 0.5 else S for t in truth]
    same = paired_bootstrap_metric(truth, good, good, macro_f1, n_boot=300)
    assert same["mean_diff"] == 0.0 and same["ci_low"] == same["ci_high"] == 0.0
    gap = paired_bootstrap_metric(truth, good, bad, macro_f1, n_boot=300)
    assert gap["mean_diff"] > 0 and gap["ci_low"] > 0
    with pytest.raises(ValueError):
        paired_bootstrap_metric([S], [S, C], [S], macro_f1)


def test_wilson_interval_behaves_at_the_extremes():
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and 0.2 < high < 0.4
    low, high = wilson_interval(5, 10)
    assert low < 0.5 < high
    assert wilson_interval(0, 0) == (0.0, 1.0)
