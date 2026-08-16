#!/usr/bin/env python3
"""Run the evaluation harness and enforce quality thresholds.

Exits non-zero when a metric falls below its threshold, so CI can use this
as a quality gate on every push -- the AI-system equivalent of a failing
unit test.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --json-out eval_report.json --markdown-out eval_report.md
"""
from __future__ import annotations

import argparse
import sys

from agentrag.config import settings
from agentrag.evaluation import check_thresholds, evaluate, format_report, to_json
from agentrag.ingest import build_corpus
from agentrag.llm import get_llm
from agentrag.vectorstore import get_vector_store


def main() -> int:
    parser = argparse.ArgumentParser(description="AgentRAG evaluation harness")
    parser.add_argument("--docs", default=settings.docs_dir)
    parser.add_argument("--vector-backend", default=settings.vector_backend, choices=["auto", "dense", "tfidf"])
    parser.add_argument("--llm-backend", default=settings.llm_backend, choices=["auto", "gemini", "extractive"])
    parser.add_argument("--json-out", help="Write the full JSON report here")
    parser.add_argument("--markdown-out", help="Write the markdown summary table here")
    parser.add_argument("--no-gate", action="store_true", help="Report metrics but always exit 0")
    args = parser.parse_args()

    corpus = build_corpus(args.docs, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    store = get_vector_store(args.vector_backend)
    store.build(corpus)
    llm = get_llm(args.llm_backend)

    print(f"Evaluating: retrieval={store.backend_name}  generation={llm.backend_name}  chunks={len(corpus)}\n")

    report = evaluate(store, llm=llm)
    markdown = format_report(report)
    print(markdown)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(to_json(report))
        print(f"\nJSON report written to {args.json_out}")

    if args.markdown_out:
        with open(args.markdown_out, "w", encoding="utf-8") as fh:
            fh.write(markdown + "\n")
        print(f"Markdown report written to {args.markdown_out}")

    failures = check_thresholds(report["metrics"])
    if failures:
        print("\nQUALITY GATE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 0 if args.no_gate else 1

    print("\nQuality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
