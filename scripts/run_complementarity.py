#!/usr/bin/env python3
"""How complementary are BM25 and dense retrieval on SciFact?

Two measurements behind the choice of hybrid as the app default:

1. Blind spots: queries where a system retrieves NONE of the gold documents in
   its top 100, and how many of those the hybrid still covers.
2. Rare-term queries: nDCG@10 on queries containing a term found in at most
   ``--max-df`` documents, with paired-bootstrap intervals.

Reads ``docs/results/scifact_per_query.json`` (committed) and the BEIR corpus;
no model inference. Writes ``docs/results/scifact_complementarity.json``.

Usage:
    python scripts/run_complementarity.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np

from agentrag.bench import data
from agentrag.bench.analysis import blind_spots, rare_term_queries
from agentrag.bench.registry import CACHE, RESULTS
from agentrag.bench.stats import paired_bootstrap
from agentrag.bench.systems import BM25Retriever

SYSTEMS = ("bm25", "bge-small", "hybrid-bge")
MAX_DFS = (3, 10)


def main() -> int:
    ds = data.load("scifact", CACHE)
    per_query = json.loads((RESULTS / "scifact_per_query.json").read_text(encoding="utf-8"))
    qids = sorted(ds.queries)

    doc_freq: Counter = Counter()
    for text in ds.corpus.values():
        doc_freq.update(set(BM25Retriever.tokenize(text)))
    query_terms = {q: BM25Retriever.tokenize(ds.queries[q]) for q in qids}

    out: dict = {"queries": len(qids), "blind_spots": {}, "rare_term": {}}

    print("Blind spots (no gold document in the top 100):")
    missed = {s: set(blind_spots({q: per_query[s][q]["recall@100"] for q in qids})) for s in per_query}
    for system in sorted(missed):
        print(f"  {system:<18} {len(missed[system])} queries")
    for weak in ("bge-small", "bm25"):
        recovered = sorted(q for q in missed[weak] if q not in missed["hybrid-bge"])
        out["blind_spots"][weak] = {
            "queries_missed": len(missed[weak]),
            "recovered_by_hybrid_bge": len(recovered),
        }
        print(f"  hybrid-bge recovers {len(recovered)} of the {len(missed[weak])} queries {weak} misses entirely")
    out["blind_spots"]["hybrid-bge"] = {"queries_missed": len(missed["hybrid-bge"])}

    for max_df in MAX_DFS:
        rare = rare_term_queries(query_terms, doc_freq, max_df)
        other = [q for q in qids if q not in set(rare)]
        block: dict = {"rare_queries": len(rare), "other_queries": len(other), "mean_ndcg@10": {}, "diffs": {}}
        print(f"\nRare-term queries (a query term occurs in <= {max_df} documents): {len(rare)} of {len(qids)}")
        for label, subset in (("rare", rare), ("other", other)):
            means = {s: float(np.mean([per_query[s][q]["ndcg@10"] for q in subset])) for s in SYSTEMS}
            block["mean_ndcg@10"][label] = means
            print(f"  {label:<6} n={len(subset):<3} " + "  ".join(f"{s}={v:.3f}" for s, v in means.items()))
        for a, b in (("hybrid-bge", "bge-small"), ("hybrid-bge", "bm25"), ("bm25", "bge-small")):
            scores_a = [per_query[a][q]["ndcg@10"] for q in rare]
            scores_b = [per_query[b][q]["ndcg@10"] for q in rare]
            test = paired_bootstrap(scores_a, scores_b)
            block["diffs"][f"{a} - {b} (rare)"] = test
            ci = f"[{test['ci_low']:+.3f}, {test['ci_high']:+.3f}]"
            print(f"  rare: {a} - {b} = {test['mean_diff']:+.3f}  95% CI {ci}")
        out["rare_term"][f"max_df_{max_df}"] = block

    (RESULTS / "scifact_complementarity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
