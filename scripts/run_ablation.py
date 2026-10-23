#!/usr/bin/env python3
"""Run retrieval systems on BEIR SciFact and report Recall@k, MRR and nDCG.

Each system's top-100 run and per-query metrics are saved under
``.bench_cache/runs/`` so later steps (fusion, significance tests, error
analysis) can reuse them without re-encoding the corpus.

Usage:
    pip install -e ".[bench]"
    python scripts/run_ablation.py --systems bm25
    python scripts/run_ablation.py --systems bm25,minilm,bge-small
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from agentrag.bench import data, metrics
from agentrag.bench.systems import BM25Retriever, DenseRetriever, Retriever

CACHE = Path(".bench_cache")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

FACTORIES = {
    "bm25": lambda: BM25Retriever(),
    "minilm": lambda: DenseRetriever("sentence-transformers/all-MiniLM-L6-v2", cache_dir=CACHE / "emb"),
    "bge-small": lambda: DenseRetriever(
        "BAAI/bge-small-en-v1.5", query_prefix=BGE_QUERY_PREFIX, cache_dir=CACHE / "emb"
    ),
}


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]


def run_system(retriever: Retriever, ds: data.BeirDataset, depth: int = 100, warmup: int = 5) -> dict:
    t0 = time.perf_counter()
    retriever.index(ds.corpus)
    index_s = time.perf_counter() - t0

    qids = list(ds.queries)
    for qid in qids[:warmup]:
        retriever.search(ds.queries[qid], k=depth)

    run: dict[str, list[str]] = {}
    latencies: list[float] = []
    for qid in qids:
        start = time.perf_counter()
        hits = retriever.search(ds.queries[qid], k=depth)
        latencies.append((time.perf_counter() - start) * 1000)
        run[qid] = [doc_id for doc_id, _ in hits]

    per_query = metrics.per_query_metrics(run, ds.qrels)
    return {
        "system": retriever.name,
        "queries": len(qids),
        "metrics": metrics.mean_metrics(per_query),
        "latency_ms": {"p50": statistics.median(latencies), "p95": percentile(latencies, 0.95)},
        "index_seconds": index_s,
        "per_query": per_query,
        "run": run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="BEIR SciFact retrieval ablation")
    parser.add_argument("--systems", default="bm25", help=f"comma-separated: {', '.join(FACTORIES)}")
    parser.add_argument("--json-out", help="Write a summary (no per-query detail) here")
    args = parser.parse_args()

    ds = data.load("scifact", CACHE)
    print(f"SciFact: {len(ds.corpus)} docs, {len(ds.queries)} test queries\n")

    summary = []
    for key in (s.strip() for s in args.systems.split(",")):
        retriever = FACTORIES[key]()
        result = run_system(retriever, ds)
        runs_dir = CACHE / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / f"{key}.json").write_text(json.dumps(result), encoding="utf-8")

        m, lat = result["metrics"], result["latency_ms"]
        print(
            f"{retriever.name:<34} nDCG@10={m['ndcg@10']:.4f}  MRR@10={m['mrr@10']:.4f}  "
            f"R@5={m['recall@5']:.4f}  R@50={m['recall@50']:.4f}  R@100={m['recall@100']:.4f}  "
            f"p50={lat['p50']:.1f}ms  p95={lat['p95']:.1f}ms  index={result['index_seconds']:.1f}s"
        )
        summary.append({k: v for k, v in result.items() if k not in ("per_query", "run")})

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
