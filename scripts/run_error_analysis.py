#!/usr/bin/env python3
"""Error analysis: the 20 SciFact queries where hybrid loses most to BM25.

Reads the cached runs from ``.bench_cache/runs/`` (produced by run_ablation.py)
and writes ``docs/results/scifact_error_analysis.{json,md}``. Grouping rules
live in ``agentrag.bench.analysis`` and are documented there.

Usage:
    python scripts/run_error_analysis.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from agentrag.bench import data
from agentrag.bench.analysis import GROUPS, classify_by_dense_rank, loss_driver, overtakers, rank_of, worst_losses

CACHE = Path(".bench_cache")
RESULTS = Path("docs/results")
WINNER, LOSER, DENSE = "bm25", "hybrid-bge", "bge-small"  # hybrid-bge fuses bm25 + bge-small


def _within(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def main() -> int:
    ds = data.load("scifact", CACHE)
    runs = {k: json.loads((CACHE / "runs" / f"{k}.json").read_text(encoding="utf-8")) for k in (WINNER, LOSER, DENSE)}
    ndcg = {k: {q: row["ndcg@10"] for q, row in runs[k]["per_query"].items()} for k in runs}

    losses, totals = worst_losses(ndcg[WINNER], ndcg[LOSER], n=20)

    # bge-small truncates input at 512 tokens; check whether that could explain any miss.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

    rows = []
    for qid, diff in losses:
        gold = [d for d, rel in ds.qrels[qid].items() if rel > 0]
        driver = loss_driver(gold, runs[WINNER]["run"][qid], runs[LOSER]["run"][qid])
        ranks = {k: rank_of(runs[k]["run"][qid], driver) for k in (WINNER, DENSE, LOSER)}
        passed = overtakers(driver, runs[WINNER]["run"][qid], runs[LOSER]["run"][qid])
        rows.append(
            {
                "qid": qid,
                "query": ds.queries[qid],
                "ndcg_diff": round(diff, 4),
                "driver_doc": driver,
                "driver_doc_start": ds.corpus[driver][:70],
                "driver_doc_tokens": len(tokenizer(ds.corpus[driver])["input_ids"]),
                "rank_bm25": ranks[WINNER],
                "rank_dense": ranks[DENSE],
                "rank_hybrid": ranks[LOSER],
                "group": classify_by_dense_rank(ranks[DENSE]),
                "overtakers": [
                    {
                        "doc": d,
                        "rank_bm25": rank_of(runs[WINNER]["run"][qid], d),
                        "rank_dense": rank_of(runs[DENSE]["run"][qid], d),
                    }
                    for d in passed
                ],
            }
        )

    counts = Counter(r["group"] for r in rows)
    overtaker_ranks = [o for r in rows if r["group"] == "consensus-demotion" for o in r["overtakers"]]
    distinct_docs = {g: len({r["driver_doc"] for r in rows if r["group"] == g}) for g in GROUPS}
    summary = {
        "comparison": f"{LOSER} vs {WINNER} on nDCG@10",
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in totals.items()},
        "group_counts": {g: counts.get(g, 0) for g in GROUPS},
        "distinct_driver_docs": len({r["driver_doc"] for r in rows}),
        "distinct_driver_docs_by_group": distinct_docs,
        "queries_whose_driver_doc_exceeds_512_tokens": sum(1 for r in rows if r["driver_doc_tokens"] > 512),
        "max_driver_doc_tokens": max(r["driver_doc_tokens"] for r in rows),
        "consensus_demotion_overtakers": len(overtaker_ranks),
        "overtakers_in_dense_top5": sum(1 for o in overtaker_ranks if _within(o["rank_dense"], 5)),
        "overtakers_in_bm25_top5": sum(1 for o in overtaker_ranks if _within(o["rank_bm25"], 5)),
    }
    (RESULTS / "scifact_error_analysis.json").write_text(
        json.dumps({"summary": summary, "queries": rows}, indent=1), encoding="utf-8"
    )

    lines = [
        f"{LOSER} vs {WINNER}: lost {totals['queries_lost']}, won {totals['queries_won']}, "
        f"tied {totals['queries_tied']} of {len(ds.queries)} queries "
        f"(sum of losses {totals['total_loss']:+.2f}, sum of gains {totals['total_gain']:+.2f} nDCG@10).",
        "",
        "| group | queries | distinct gold docs |",
        "| --- | --- | --- |",
        *[f"| {g} | {counts.get(g, 0)} | {distinct_docs[g]} |" for g in GROUPS],
        "",
        "| query id | nDCG@10 change | gold-doc rank: BM25 / dense / hybrid | group | query |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        ranks = " / ".join(
            str(r[k]) if r[k] is not None else ">100" for k in ("rank_bm25", "rank_dense", "rank_hybrid")
        )
        lines.append(f"| {r['qid']} | {r['ndcg_diff']:+.3f} | {ranks} | {r['group']} | {r['query']} |")
    (RESULTS / "scifact_error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
