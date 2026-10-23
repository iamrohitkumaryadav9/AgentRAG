"""Retrieval benchmarking: BEIR SciFact ablation of retrieval strategies.

Everything here is optional tooling. ``metrics`` and ``stats`` are pure
numpy and safe to import anywhere (CI unit-tests them); ``systems`` imports
its heavy dependencies (faiss, sentence-transformers, rank_bm25) lazily and
needs ``pip install -e ".[bench]"``.
"""
