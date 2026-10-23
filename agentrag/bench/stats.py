"""Paired significance testing for per-query IR metrics.

Two systems are scored on the *same* queries, so the right unit is the
per-query difference. A paired bootstrap resamples queries with replacement,
recomputes the mean difference each time, and reads the confidence interval
off the resulting distribution. Report the interval, not just the p-value:
it says how large the gain plausibly is, which a p-value alone does not.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap the mean of ``a - b`` over paired per-query scores.

    Returns the observed mean difference, a percentile confidence interval,
    and a two-sided bootstrap p-value (twice the smaller tail mass on either
    side of zero, floored at 1 / n_boot since that is the test's resolution).
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size == 0:
        raise ValueError("a and b must be equal-length, non-empty 1-D sequences")

    diffs = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boot_means = diffs[idx].mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(boot_means, [tail, 1.0 - tail])
    p = 2.0 * min(float((boot_means <= 0).mean()), float((boot_means >= 0).mean()))
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(min(1.0, max(p, 1.0 / n_boot))),
    }


def paired_bootstrap_metric(
    y_true: Sequence,
    pred_a: Sequence,
    pred_b: Sequence,
    metric,
    n_boot: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap ``metric(y_true, pred_a) - metric(y_true, pred_b)`` by resampling items.

    For metrics that are not per-item means (macro-F1, Cohen's kappa, per-class recall), where the
    per-item difference trick in ``paired_bootstrap`` does not apply. Both systems are re-scored on
    the same resampled items, so the interval reflects paired uncertainty.
    """
    n = len(y_true)
    if not (n == len(pred_a) == len(pred_b)) or n == 0:
        raise ValueError("y_true, pred_a and pred_b must be equal-length and non-empty")
    truth, a, b = np.asarray(y_true, dtype=object), np.asarray(pred_a, dtype=object), np.asarray(pred_b, dtype=object)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = metric(truth[idx], a[idx]) - metric(truth[idx], b[idx])
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(diffs, [tail, 1.0 - tail])
    p = 2.0 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return {
        "mean_diff": float(metric(truth, a) - metric(truth, b)),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(min(1.0, max(p, 1.0 / n_boot))),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion; well behaved for small counts and rates near 0 or 1."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls family-wise error rate).

    Comparing many system pairs at once inflates false positives; Holm is
    uniformly more powerful than plain Bonferroni and just as safe.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted
