#!/usr/bin/env python3
"""Paired bootstrap significance tests on per-query nDCG@10.

Reads ``docs/results/scifact_per_query.json`` (written by run_ablation.py),
tests every pair of systems, and Holm-adjusts p-values across ALL pairs so
the headline comparisons are not cherry-picked from a larger family. A short
list of pre-declared comparisons is printed as a table; the full pairwise
matrix goes to ``docs/results/scifact_significance.json``.

Usage:
    python scripts/run_significance.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

from agentrag.bench.stats import holm_adjust, paired_bootstrap

RESULTS = Path("docs/results")
METRIC = "ndcg@10"

# Declared up front, not chosen after looking at the numbers.
PRIMARY = [
    ("bm25", "tfidf"),
    ("minilm", "bm25"),
    ("bge-small", "bm25"),
    ("bge-small", "minilm"),
    ("hybrid-minilm", "bm25"),
    ("hybrid-minilm", "minilm"),
    ("hybrid-bge", "bm25"),
    ("hybrid-bge", "bge-small"),
    ("hybrid-bge-rerank", "hybrid-bge"),
    ("hybrid-bge-rerank", "bge-small"),
]


def main() -> int:
    per_query = json.loads((RESULTS / "scifact_per_query.json").read_text(encoding="utf-8"))
    systems = sorted(per_query)
    qids = sorted(next(iter(per_query.values())))
    scores = {s: [per_query[s][q][METRIC] for q in qids] for s in systems}

    pairs = list(itertools.combinations(systems, 2))
    tests = {}
    for a, b in pairs:
        tests[(a, b)] = paired_bootstrap(scores[a], scores[b])
    adjusted = holm_adjust([tests[p]["p_value"] for p in pairs])
    for pair, adj in zip(pairs, adjusted, strict=True):
        tests[pair]["p_holm"] = adj

    def lookup(a: str, b: str) -> dict:
        if (a, b) in tests:
            return tests[(a, b)]
        flipped = tests[(b, a)]
        return {
            "mean_diff": -flipped["mean_diff"],
            "ci_low": -flipped["ci_high"],
            "ci_high": -flipped["ci_low"],
            "p_value": flipped["p_value"],
            "p_holm": flipped["p_holm"],
        }

    print(f"Paired bootstrap on per-query {METRIC}: {len(qids)} queries, 10,000 resamples, "
          f"Holm-adjusted across {len(pairs)} pairs\n")
    print("| A vs B | mean diff (A-B) | 95% CI | p | p (Holm) |")
    print("| --- | --- | --- | --- | --- |")
    for a, b in PRIMARY:
        if a in scores and b in scores:
            t = lookup(a, b)
            print(f"| {a} vs {b} | {t['mean_diff']:+.4f} | [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] "
                  f"| {t['p_value']:.4f} | {t['p_holm']:.4f} |")

    out = {f"{a} vs {b}": t for (a, b), t in tests.items()}
    (RESULTS / "scifact_significance.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
