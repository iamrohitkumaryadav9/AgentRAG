#!/usr/bin/env python3
"""Standalone helper to (re)build the index and print a quick sanity report.

Useful in CI or as a pre-warm step before starting the FastAPI service.
"""
from __future__ import annotations

import argparse

from agentrag.config import settings
from agentrag.ingest import build_corpus
from agentrag.vectorstore import get_vector_store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default=settings.docs_dir)
    parser.add_argument("--backend", default=settings.vector_backend, choices=["auto", "dense", "tfidf"])
    args = parser.parse_args()

    corpus = build_corpus(args.docs, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    store = get_vector_store(args.backend)
    store.build(corpus)

    print(f"Backend: {store.backend_name}")
    print(f"Documents chunked into {len(corpus)} chunks from '{args.docs}'")
    sample = store.search("What is this project about?", k=3)
    print("\nSanity check retrieval for 'What is this project about?':")
    for r in sample:
        print(f"  [{r.score:.3f}] {r.chunk.source}#{r.chunk.position}: {r.chunk.text[:100]}...")


if __name__ == "__main__":
    main()
