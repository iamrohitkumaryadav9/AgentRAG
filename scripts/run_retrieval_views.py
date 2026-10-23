#!/usr/bin/env python3
"""Retrieval quality on two views of the SciFact test set, with no model inference.

* all 300 claims (BEIR-comparable; includes the 112 NOT_ENOUGH_INFO claims, whose
  "relevant" document is the abstract the claim was written about)
* the 188 SUPPORT + CONTRADICT claims (a gold evidence document exists)

Relevance comes from the BEIR qrels. Reads the cached top-100 runs written by
``run_ablation.py`` plus the original SciFact release for the claim labels, and
writes ``docs/results/scifact_views.json``.

Usage:
    python scripts/run_retrieval_views.py
"""
from __future__ import annotations

import json
import sys

import numpy as np

from agentrag.bench import data as beir_data
from agentrag.bench import metrics
from agentrag.bench.registry import CACHE, RESULTS
from agentrag.bench.stats import paired_bootstrap
from agentrag.verify import data as sf_data

SYSTEMS = ("tfidf", "bm25", "minilm", "bge-small", "hybrid-minilm", "hybrid-bge", "hybrid-bge-rerank")
COLUMNS = ("ndcg@10", "mrr@10", "recall@5", "recall@100", "hit@1", "hit@5", "hit@10",
           "context_precision@4", "context_recall@4")
COMPARISONS = (
    ("hybrid-bge", "bm25"),
    ("hybrid-bge", "bge-small"),
    ("bge-small", "bm25"),
    ("hybrid-bge-rerank", "hybrid-bge"),
)


def main() -> int:
    beir = beir_data.load("scifact", CACHE)
    sf = sf_data.load(CACHE)
    parity = sf_data.check_beir_parity(sf, beir.queries, beir.corpus)
    print(f"parity with BEIR confirmed: {parity}\n")

    label_of = {c.id: c.label for c in sf.dev}
    views = {
        "all_300": sorted(beir.queries),
        "evidence_188": sorted(q for q in beir.queries if label_of[q] != sf_data.NEI),
    }
    evidence_docs = {c.id: c.evidence_docs for c in sf.dev}

    runs = {s: json.loads((CACHE / "runs" / f"{s}.json").read_text(encoding="utf-8"))["run"] for s in SYSTEMS}
    per_query = {
        s: {q: metrics.view_metrics(runs[s][q], beir.qrels[q]) for q in beir.queries} for s in SYSTEMS
    }

    out: dict = {"parity": parity, "views": {}}
    for view, qids in views.items():
        table = {s: {c: float(np.mean([per_query[s][q][c] for q in qids])) for c in COLUMNS} for s in SYSTEMS}
        block: dict = {"claims": len(qids), "systems": table, "comparisons_ndcg@10": {}}
        if view == "evidence_188":
            # top-5 contains at least one *evidence* document: the ceiling for verdict accuracy
            block["evidence_doc_hit@5"] = {
                s: float(np.mean([bool(evidence_docs[q] & set(runs[s][q][:5])) for q in qids])) for s in SYSTEMS
            }
        for a, b in COMPARISONS:
            block["comparisons_ndcg@10"][f"{a} - {b}"] = paired_bootstrap(
                [per_query[a][q]["ndcg@10"] for q in qids], [per_query[b][q]["ndcg@10"] for q in qids]
            )
        out["views"][view] = block

        print(f"=== {view}: {len(qids)} claims")
        print("| System | " + " | ".join(COLUMNS) + (" | evidence-doc hit@5 |" if view == "evidence_188" else " |"))
        print("| --- |" + " --- |" * (len(COLUMNS) + (1 if view == "evidence_188" else 0)))
        for s in SYSTEMS:
            extra = f" {block['evidence_doc_hit@5'][s]:.3f} |" if view == "evidence_188" else ""
            print(f"| {s} | " + " | ".join(f"{table[s][c]:.3f}" for c in COLUMNS) + f" |{extra}")
        for name, t in block["comparisons_ndcg@10"].items():
            print(f"  nDCG@10 {name}: {t['mean_diff']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}]")
        print()

    (RESULTS / "scifact_views.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
