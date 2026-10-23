"""Metrics for the claim-verification experiments (no third-party ML dependency).

Verdict quality: macro-F1 and Cohen's kappa against the human labels, with
per-class precision / recall (contradiction recall is the number to watch,
read together with its precision so it cannot be gamed by over-predicting).
Critic behaviour: how often it flags, how often a flag is right, and the
fix-versus-harm accounting of the retry loop. Faithfulness and evidence
overlap quantify whether the cited sentences really carry the verdict.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence

from .data import LABELS


def confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, dict[str, int]]:
    matrix = {t: {p: 0 for p in LABELS} for t in LABELS}
    for truth, pred in zip(y_true, y_pred, strict=True):
        matrix[truth][pred] += 1
    return matrix


def per_class(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, dict[str, float]]:
    matrix = confusion(y_true, y_pred)
    out = {}
    for label in LABELS:
        tp = matrix[label][label]
        predicted = sum(matrix[t][label] for t in LABELS)
        actual = sum(matrix[label].values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}
    return out


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    classes = per_class(y_true, y_pred)
    return sum(c["f1"] for c in classes.values()) / len(LABELS)


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)


def cohens_kappa(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Unweighted Cohen's kappa: agreement beyond what the two label distributions predict by chance."""
    n = len(y_true)
    observed = accuracy(y_true, y_pred)
    truth, pred = Counter(y_true), Counter(y_pred)
    expected = sum(truth[label] * pred[label] for label in LABELS) / (n * n)
    return 1.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)


def contradiction_recall(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return per_class(y_true, y_pred)["CONTRADICT"]["recall"]


def summarize(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    return {
        "n": len(y_true),
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
        "kappa": cohens_kappa(y_true, y_pred),
        "per_class": per_class(y_true, y_pred),
        "confusion": confusion(y_true, y_pred),
    }


def retry_stats(gold: Sequence[str], initial: Sequence[str], flagged: Sequence[bool], final: Sequence[str]) -> dict:
    """Accounting for the critic + retry loop, per claim.

    flagged   the critic rejected the first answer
    A flag is *right* when the first answer really was wrong. A retry *fixes* a claim when it was wrong and
    the final answer is right, and *harms* it when it was right and the final answer is wrong.
    """
    n = len(gold)
    wrong = [i != g for i, g in zip(initial, gold, strict=True)]
    final_wrong = [f != g for f, g in zip(final, gold, strict=True)]
    n_flagged = sum(flagged)
    flagged_and_wrong = sum(f and w for f, w in zip(flagged, wrong, strict=True))
    fixed = sum(f and w and not fw for f, w, fw in zip(flagged, wrong, final_wrong, strict=True))
    harmed = sum(f and not w and fw for f, w, fw in zip(flagged, wrong, final_wrong, strict=True))
    return {
        "claims": n,
        "flagged": n_flagged,
        "flag_rate": n_flagged / n if n else 0.0,
        "initial_wrong": sum(wrong),
        "critic_precision": flagged_and_wrong / n_flagged if n_flagged else None,  # flagged claims that were wrong
        "critic_recall": flagged_and_wrong / sum(wrong) if sum(wrong) else None,  # wrong claims that were flagged
        "fixed": fixed,
        "harmed": harmed,
        "fix_rate": fixed / flagged_and_wrong if flagged_and_wrong else None,  # wrong+flagged that ended right
        "harm_rate": harmed / (n_flagged - flagged_and_wrong) if n_flagged > flagged_and_wrong else None,
        "net_fixed_per_100_claims": 100.0 * (fixed - harmed) / n if n else 0.0,
    }


def faithfulness(flags: Sequence[bool | None]) -> dict:
    """Share of SUPPORT / CONTRADICT verdicts whose cited sentences entail the verdict.

    ``flags`` holds one entry per verdict: True / False from the blind polarity check, or None when the
    notion does not apply (NOT_ENOUGH_INFO, no checker); those are excluded from the denominator.
    """
    defined = [f for f in flags if f is not None]
    return {"faithful": sum(defined), "judged": len(defined), "rate": sum(defined) / len(defined) if defined else None}


def evidence_prf(pairs: Sequence[tuple[set, set]]) -> dict:
    """Micro-averaged sentence-level precision / recall / F1 of cited vs gold rationale sentences.

    ``pairs`` holds (predicted, gold) sets of (doc_id, sentence_index), only for claims with gold evidence;
    a claim that cites nothing contributes its gold sentences to the recall denominator.
    """
    hit = sum(len(p & g) for p, g in pairs)
    predicted = sum(len(p) for p, _ in pairs)
    gold = sum(len(g) for _, g in pairs)
    precision = hit / predicted if predicted else 0.0
    recall = hit / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "claims": len(pairs)}


def checker_validation(expected: Sequence[str], predicted: Sequence[str]) -> dict:
    """How well a polarity checker recovers a known relation (labels: entails / contradicts / neutral)."""
    pairs = list(zip(expected, predicted, strict=True))
    per = {}
    for c in sorted(set(expected)):
        n_c = sum(e == c for e, _ in pairs)
        per[c] = {"n": n_c, "recall": sum(e == c and p == c for e, p in pairs) / n_c}
    return {"n": len(pairs), "accuracy": sum(e == p for e, p in pairs) / len(pairs), "per_class": per}


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_usage(per_claim: Sequence[Mapping[str, float]], keys: Sequence[str]) -> dict[str, float]:
    return {k: mean([row[k] for row in per_claim]) for k in keys}


MetricFn = Callable[[Sequence[str], Sequence[str]], float]
