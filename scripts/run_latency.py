#!/usr/bin/env python3
"""Retrieval latency on SciFact in three execution modes.

    seq-1thread      one query at a time, PyTorch and FAISS limited to 1 thread
    seq-default      one query at a time, library default threads (what serving
                     a single request looks like)
    batched-default  all queries in one batched call, default threads
                     (offline / throughput; reported as amortised ms per query)

nDCG@10 is computed in every mode to show batching does not change results.
Only PyTorch and FAISS thread counts are controlled: NumPy/SciPy BLAS threads
and the pure-Python BM25 loop are not, so BM25/TF-IDF numbers barely move
across modes. The cross-encoder is slow, so it runs on a smaller query subset.

Usage:
    python scripts/run_latency.py
    python scripts/run_latency.py --systems bm25,bge-small --rerank-queries 20
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

from agentrag.bench import data, metrics
from agentrag.bench.registry import CACHE, FACTORIES, RESULTS, make_getter, percentile

DEFAULT_SYSTEMS = "tfidf,bm25,minilm,bge-small,hybrid-minilm,hybrid-bge,hybrid-bge-rerank"


def set_threads(torch_threads: int, faiss_threads: int) -> None:
    import faiss
    import torch

    torch.set_num_threads(torch_threads)
    faiss.omp_set_num_threads(faiss_threads)


def mean_ndcg(run: dict[str, list[str]], qrels: dict) -> float:
    return metrics.mean_metrics(metrics.per_query_metrics(run, qrels))["ndcg@10"]


def measure_sequential(retriever, queries: list[str], qids: list[str], qrels: dict) -> dict:
    for q in queries[:5]:
        retriever.search(q, k=100)
    times, run = [], {}
    for qid, q in zip(qids, queries, strict=True):
        start = time.perf_counter()
        hits = retriever.search(q, k=100)
        times.append((time.perf_counter() - start) * 1000)
        run[qid] = [d for d, _ in hits]
    return {
        "p50_ms": statistics.median(times),
        "p95_ms": percentile(times, 0.95),
        "ndcg@10": mean_ndcg(run, qrels),
    }


def measure_batched(retriever, queries: list[str], qids: list[str], qrels: dict) -> dict:
    retriever.search_batch(queries[:5], k=100)
    start = time.perf_counter()
    results = retriever.search_batch(queries, k=100)
    total_ms = (time.perf_counter() - start) * 1000
    run = {qid: [d for d, _ in hits] for qid, hits in zip(qids, results, strict=True)}
    return {
        "ms_per_query": total_ms / len(queries),
        "queries_per_second": len(queries) / (total_ms / 1000),
        "ndcg@10": mean_ndcg(run, qrels),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SciFact retrieval latency, three execution modes")
    parser.add_argument("--systems", default=DEFAULT_SYSTEMS, help=f"comma-separated: {', '.join(FACTORIES)}")
    parser.add_argument("--rerank-queries", type=int, default=30, help="query subset size for cross-encoder systems")
    args = parser.parse_args()

    import faiss
    import torch

    default_torch, default_faiss = torch.get_num_threads(), faiss.omp_get_max_threads()
    ds = data.load("scifact", CACHE)
    all_qids = list(ds.queries)
    get = make_getter()

    env = {
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "default_torch_threads": default_torch,
        "default_faiss_threads": default_faiss,
    }
    print(f"env: {env}\n")

    out: dict = {"env": env, "systems": {}}
    for key in (s.strip() for s in args.systems.split(",")):
        retriever = get(key)
        if not retriever.indexed:
            retriever.index(ds.corpus)

        n = args.rerank_queries if "rerank" in key else len(all_qids)
        qids = all_qids[:n]
        queries = [ds.queries[q] for q in qids]
        qrels = {q: ds.qrels[q] for q in qids}

        set_threads(1, 1)
        one = measure_sequential(retriever, queries, qids, qrels)
        set_threads(default_torch, default_faiss)
        seq = measure_sequential(retriever, queries, qids, qrels)
        batched = measure_batched(retriever, queries, qids, qrels)

        out["systems"][key] = {"queries": n, "seq_1thread": one, "seq_default": seq, "batched_default": batched}
        print(
            f"{key:<18} n={n:<3} "
            f"seq-1thread p50={one['p50_ms']:8.1f}ms | seq-default p50={seq['p50_ms']:8.1f}ms | "
            f"batched {batched['ms_per_query']:8.1f}ms/query ({batched['queries_per_second']:.1f} q/s) | "
            f"nDCG@10 {one['ndcg@10']:.4f}/{seq['ndcg@10']:.4f}/{batched['ndcg@10']:.4f}"
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "scifact_latency.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"systems": {}}
    existing["env"] = env
    existing["systems"].update(out["systems"])
    path.write_text(json.dumps(existing, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
