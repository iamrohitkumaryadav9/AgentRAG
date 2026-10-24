"""Experiment plumbing for the claim-verification study: pure, offline-testable pieces.

The run script (scripts/run_verification.py) wires these to real retrieval, the
Gemini client and the NLI model. Everything here takes plain data.
"""
from __future__ import annotations

import random
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence

from .critic import resolve_citations
from .data import CONTRADICT, SUPPORT, Claim, Doc, SciFact
from .graph import ClaimOutcome
from .metrics import evidence_prf, faithfulness, retry_stats, summarize
from .nli import Probs, decide_relation
from .schemas import Relation, Verdict

TAU_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)
GOLD_RELATION = {SUPPORT: Relation.ENTAILS, CONTRADICT: Relation.CONTRADICTS}


def docs_for(sf: SciFact, doc_ids: Sequence[str]) -> list[Doc]:
    return [sf.docs[d] for d in doc_ids if d in sf.docs]


def oracle_docs(sf: SciFact, claim: Claim) -> list[Doc]:
    """Gold abstracts: the evidence documents for SUPPORT / CONTRADICT claims, the cited ones for NOT_ENOUGH_INFO."""
    ids = sorted(claim.evidence_docs) if claim.evidence_docs else list(claim.cited_doc_ids)
    return docs_for(sf, ids)


def gold_pairs(claim: Claim) -> set[tuple[str, int]]:
    return {(doc_id, i) for doc_id, indices in claim.rationale.items() for i in indices}


def predicted_pairs(outcome: ClaimOutcome, docs: Sequence[Doc]) -> set[tuple[str, int]]:
    last = outcome.attempts[-1]
    if last.output is None:
        return set()
    valid, _ = resolve_citations(last.output, docs)
    return set(valid)


def attempt_record(attempt) -> dict:
    report = attempt.report
    return {
        "status": attempt.judge_status,
        "verdict": attempt.verdict.value,
        "rationale": attempt.output.rationale if attempt.output else "",
        "citations": [c.model_dump() for c in attempt.output.citations] if attempt.output else [],
        "critic": None if report is None else {
            "passed": report.passed,
            "rule_failures": report.rule_failures,
            "relation": report.relation.value if report.relation else None,
            "expected": report.expected.value if report.expected else None,
            "polarity_ok": report.polarity_ok,
            "faithful": report.faithful,
            "checker_status": report.checker_status,
            "feedback": report.feedback,
        },
    }


def claim_record(claim: Claim, outcome: ClaimOutcome, retrieved: Sequence[str]) -> dict:
    return {
        "id": claim.id,
        "gold": claim.label,
        "claim": claim.text,
        "retrieved": list(retrieved),
        "initial": outcome.initial_verdict.value,
        "final_keep": outcome.final_keep.value,
        "final_abstain": outcome.final_abstain.value,
        "flagged": outcome.flagged,
        "attempts": [attempt_record(a) for a in outcome.attempts],
    }


def _percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))] if ordered else 0.0


def usage_summary(outcomes: Sequence[ClaimOutcome]) -> dict:
    usages = [o.usage for o in outcomes]
    n = len(usages) or 1
    latencies = [u.latency_ms for u in usages]
    return {
        "claims": len(usages),
        "api_calls_total": sum(u.api_calls for u in usages),
        "model_calls_total": sum(u.model_calls for u in usages),
        "model_calls_per_claim": sum(u.model_calls for u in usages) / n,
        "prompt_tokens_per_claim": sum(u.prompt_tokens for u in usages) / n,
        "output_tokens_per_claim": sum(u.output_tokens for u in usages) / n,
        "thought_tokens_per_claim": sum(u.thought_tokens for u in usages) / n,
        "paid_equivalent_usd_total": sum(u.cost_usd for u in usages),
        "paid_equivalent_usd_per_claim": sum(u.cost_usd for u in usages) / n,
        "api_latency_ms_per_claim_mean": statistics.fmean(latencies) if latencies else 0.0,
        "api_latency_ms_per_claim_p50": _percentile(latencies, 0.5),
        "api_latency_ms_per_claim_p95": _percentile(latencies, 0.95),
        "attempts_mean": sum(len(o.attempts) for o in outcomes) / n,
    }


def evaluate_arm(claims: Sequence[Claim], outcomes: Sequence[ClaimOutcome], docs: Mapping[str, Sequence[Doc]]) -> dict:
    """Every number reported for one arm, computed from its per-claim outcomes."""
    gold = [c.label for c in claims]
    initial = [o.initial_verdict.value for o in outcomes]
    keep = [o.final_keep.value for o in outcomes]
    abstain = [o.final_abstain.value for o in outcomes]
    flagged = [o.flagged for o in outcomes]
    evidence_pairs = [
        (predicted_pairs(o, docs[c.id]), gold_pairs(c)) for c, o in zip(claims, outcomes, strict=True) if c.rationale
    ]

    def faith(index: int) -> list[bool | None]:
        return [o.attempts[index].report.faithful if o.attempts[index].report else None for o in outcomes]

    return {
        "initial": summarize(gold, initial),
        "final_keep": summarize(gold, keep),
        "final_abstain": summarize(gold, abstain),
        "retry_keep": retry_stats(gold, initial, flagged, keep),
        "retry_abstain": retry_stats(gold, initial, flagged, abstain),
        "faithfulness_initial": faithfulness(faith(0)),
        "faithfulness_final": faithfulness([o.attempts[-1].report.faithful if o.attempts[-1].report else None
                                            for o in outcomes]),
        "evidence": evidence_prf(evidence_pairs),
        "first_attempt_status": dict(Counter(o.attempts[0].judge_status for o in outcomes)),
        "final_critic_rejections": sum(not o.final_passed for o in outcomes),
        "usage": usage_summary(outcomes),
    }


# -- NLI-only baseline and threshold tuning ------------------------------------------------------


def nli_only_verdict(sentence_probs: Sequence[Probs], tau: float) -> Verdict:
    """Verdict from sentence-level NLI alone: the strongest entailing / contradicting sentence decides."""
    if not sentence_probs:
        return Verdict.NOT_ENOUGH_INFO
    entail = max(p["entailment"] for p in sentence_probs)
    contra = max(p["contradiction"] for p in sentence_probs)
    if entail >= tau and entail > contra:
        return Verdict.SUPPORT
    if contra >= tau and contra > entail:
        return Verdict.CONTRADICT
    return Verdict.NOT_ENOUGH_INFO


def _macro_f1_generic(truth: Sequence, pred: Sequence) -> float:
    scores = []
    for cls in sorted(set(truth) | set(pred), key=str):
        tp = sum(t == cls and p == cls for t, p in zip(truth, pred, strict=True))
        fp = sum(t != cls and p == cls for t, p in zip(truth, pred, strict=True))
        fn = sum(t == cls and p != cls for t, p in zip(truth, pred, strict=True))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(scores) / len(scores)


def tune_tau(samples: Sequence[tuple[Probs, Relation]], grid: Sequence[float] = TAU_GRID) -> dict:
    """Pick the (entailment, contradiction) thresholds that maximise 3-way relation macro-F1.

    Searched jointly over ``grid`` x ``grid`` on labelled (probs, relation) pairs. Ties resolve to the *higher*
    thresholds: the more conservative critic, which flags fewer correct answers.
    """
    truth = [relation for _, relation in samples]
    table = {
        (te, tc): _macro_f1_generic(truth, [decide_relation(p, te, tc) for p, _ in samples])
        for te in grid
        for tc in grid
    }
    best = max(table, key=lambda pair: (round(table[pair], 12), pair))
    per_class = {}
    predicted = [decide_relation(p, *best) for p, _ in samples]
    for relation in Relation:
        n = sum(t is relation for t in truth)
        hits = sum(t is relation and p is relation for t, p in zip(truth, predicted, strict=True))
        per_class[relation.value] = {"n": n, "recall": hits / max(1, n)}
    return {
        "tau_entail": best[0], "tau_contradict": best[1], "macro_f1": table[best], "n": len(samples),
        "per_class_recall": per_class, "grid": {f"{te},{tc}": v for (te, tc), v in table.items()},
    }


def tune_nli_only_tau(
    samples: Sequence[tuple[Sequence[Probs], str]], grid: Sequence[float] = TAU_GRID
) -> dict:
    """Same for the NLI-only verdict baseline: samples are (sentence probabilities, gold label)."""
    truth = [label for _, label in samples]
    table = {tau: _macro_f1_generic(truth, [nli_only_verdict(p, tau).value for p, _ in samples]) for tau in grid}
    best = max(grid, key=lambda tau: (round(table[tau], 12), tau))
    return {"tau": best, "macro_f1": table[best], "grid": {str(t): v for t, v in table.items()}, "n": len(samples)}


# -- validating the polarity checkers against human-annotated evidence ----------------------------


def validation_items(sf: SciFact, claims: Sequence[Claim], seed: int = 0) -> list[dict]:
    """Known-answer items for a polarity checker.

    gold      a claim's human-marked rationale sentences with the claim itself: the checker should say
              "entails" for SUPPORT claims and "contradicts" for CONTRADICT claims.
    shuffled  the same sentences paired with a different claim: the checker should say "neutral"
              (negative control for over-eager entailment).
    """
    rng = random.Random(seed)
    with_evidence = [c for c in claims if c.rationale]
    items = []
    for claim in with_evidence:
        sentences = [sf.docs[d].sentences[i] for d in sorted(claim.rationale) for i in sorted(claim.rationale[d])]
        items.append({"id": claim.id, "kind": "gold", "claim": claim.text, "sentences": sentences,
                      "expected": GOLD_RELATION[claim.label].value})
        others = [c for c in with_evidence if c.id != claim.id and not (c.evidence_docs & claim.evidence_docs)]
        other = rng.choice(others)
        items.append({"id": claim.id, "kind": "shuffled", "claim": other.text, "sentences": sentences,
                      "expected": Relation.NEUTRAL.value})
    return items
