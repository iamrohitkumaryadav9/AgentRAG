#!/usr/bin/env python3
"""Run retrieval systems on BEIR SciFact and report Recall@k, MRR and nDCG.

Each system's top-100 run is cached under ``.bench_cache/runs/`` (git-ignored)
so fusion, significance tests and error analysis can reuse it. Compact,
committable results (means, latency, per-query metrics) are merged into
``docs/results/`` so a partial run never clobbers other systems' entries.

Usage:
    pip install -e ".[bench]"
    python scripts/run_ablation.py --systems bm25
    python scripts/run_ablation.py --systems tfidf,bm25,minilm,bge-small,hybrid-minilm,hybrid-bge,hybrid-bge-rerank
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from agentrag.bench import data, metrics
from agentrag.bench.systems import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRRFRetriever,
    Retriever,
    TfidfRetriever,
)

CACHE = Path(".bench_cache")
RESULTS = Path("docs/results")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Registry: key -> factory taking `get`, which returns the shared instance of
# another system so a component is indexed once however many systems use it.
FACTORIES = {
    "tfidf": lambda get: TfidfRetriever(),
    "bm25": lambda get: BM25Retriever(),
    "minilm": lambda get: DenseRetriever("sentence-transformers/all-MiniLM-L6-v2", cache_dir=CACHE / "emb"),
    "bge-small": lambda get: DenseRetriever(
        "BAAI/bge-small-en-v1.5", query_prefix=BGE_QUERY_PREFIX, cache_dir=CACHE / "emb"
    ),
    "hybrid-minilm": lambda get: HybridRRFRetriever([get("bm25"), get("minilm")]),
    "hybrid-bge": lambda get: HybridRRFRetriever([get("bm25"), get("bge-small")]),
    "hybrid-bge-rerank": lambda get: CrossEncoderReranker(get("hybrid-bge")),
}


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]


def run_system(retriever: Retriever, ds: data.BeirDataset, depth: int = 100, warmup: int = 5) -> dict:
    t0 = time.perf_counter()
    if not retriever.indexed:
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


def _merge_json(path: Path, update: dict) -> None:
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing.update(update)
    path.write_text(json.dumps(existing, indent=1, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="BEIR SciFact retrieval ablation")
    parser.add_argument("--systems", default="bm25", help=f"comma-separated: {', '.join(FACTORIES)}")
    args = parser.parse_args()

    ds = data.load("scifact", CACHE)
    print(f"SciFact: {len(ds.corpus)} docs, {len(ds.queries)} test queries\n")

    instances: dict[str, Retriever] = {}

    def get(key: str) -> Retriever:
        if key not in instances:
            instances[key] = FACTORIES[key](get)
        return instances[key]

    (CACHE / "runs").mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    for key in (s.strip() for s in args.systems.split(",")):
        retriever = get(key)
        result = run_system(retriever, ds)
        (CACHE / "runs" / f"{key}.json").write_text(json.dumps(result), encoding="utf-8")

        summary = {k: v for k, v in result.items() if k not in ("per_query", "run")}
        _merge_json(RESULTS / "scifact_summary.json", {key: summary})
        _merge_json(RESULTS / "scifact_per_query.json", {key: result["per_query"]})

        m, lat = result["metrics"], result["latency_ms"]
        print(
            f"{key:<18} nDCG@10={m['ndcg@10']:.4f}  MRR@10={m['mrr@10']:.4f}  "
            f"R@5={m['recall@5']:.4f}  R@50={m['recall@50']:.4f}  R@100={m['recall@100']:.4f}  "
            f"p50={lat['p50']:.1f}ms  p95={lat['p95']:.1f}ms"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
