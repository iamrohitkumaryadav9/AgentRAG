#!/usr/bin/env python3
"""Terminal demo: build the index (if needed) and answer one question.

Usage:
    python cli.py "What is retrieval-augmented generation?"
    python cli.py --docs data/sample_docs "How do guardrails work?"
"""
from __future__ import annotations

import argparse
import sys

from agentrag.config import settings
from agentrag.graph import run_query
from agentrag.ingest import build_corpus
from agentrag.llm import get_llm
from agentrag.vectorstore import get_vector_store


def main() -> int:
    parser = argparse.ArgumentParser(description="AgentRAG CLI demo")
    parser.add_argument("question", help="Question to ask the knowledge base")
    parser.add_argument("--docs", default=settings.docs_dir, help="Directory of documents to index")
    parser.add_argument("--vector-backend", default=settings.vector_backend, choices=["auto", "dense", "tfidf"])
    parser.add_argument("--llm-backend", default=settings.llm_backend, choices=["auto", "gemini", "extractive"])
    args = parser.parse_args()

    print(f"Indexing '{args.docs}' ...")
    corpus = build_corpus(args.docs, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    store = get_vector_store(args.vector_backend)
    store.build(corpus)
    llm = get_llm(args.llm_backend)
    print(f"Indexed {len(corpus)} chunks  |  retrieval={store.backend_name}  generation={llm.backend_name}\n")

    result = run_query(store, args.question, llm=llm)

    print(f"Q: {args.question}\n")
    print(f"A: {result['answer']}\n")

    payload = result.get("payload")
    if payload:
        print(f"Confidence: {payload.confidence}")
        if payload.citations:
            cites = ", ".join(f"{c.source}#{c.position}" for c in payload.citations)
            print(f"Citations: {cites}")

    guardrail = result.get("guardrail")
    if guardrail:
        status = "PASSED" if guardrail.passed else "FAILED"
        print(f"Guardrail: {status}  (grounding_score={guardrail.grounding_score}, {guardrail.reason})")
    if result.get("pii_flags"):
        print(f"PII flags: {result['pii_flags']}")
    if result.get("bad_citations"):
        print(f"Fabricated citations: {result['bad_citations']}")
    print(f"Attempts: {result.get('attempts', 0)}")

    print("\nSources:")
    for r in result.get("retrieved", []):
        print(f"  - {r.chunk.source} (score={r.score:.3f})")

    print("\nTrace:")
    for step in result.get("trace", []):
        print(f"  * {step}")

    metrics = result.get("metrics")
    if metrics:
        print(f"\nTelemetry (total {metrics.total_ms} ms, ~{metrics.est_tokens} tokens):")
        for span in metrics.spans:
            print(f"  * {span.name}: {span.duration_ms} ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
