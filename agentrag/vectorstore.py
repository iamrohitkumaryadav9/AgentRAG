"""Pluggable vector store backends.

Two implementations share one interface:

* ``DenseVectorStore``  - sentence-transformer embeddings + FAISS cosine search.
  This is the "real" production path (semantic retrieval), used whenever a
  sentence-transformers model can be loaded (requires model weights, which
  are fetched from Hugging Face on first run).
* ``TfidfVectorStore``  - scikit-learn TF-IDF + cosine similarity, pure
  numpy/sklearn, no model download required. Used automatically whenever the
  dense backend can't be initialised (offline environment, no cached model,
  etc.), so the pipeline never hard-fails just because of network access.

``get_vector_store(backend="auto")`` picks the best available backend at
runtime -- a small but deliberate resilience pattern: a production RAG
service should degrade gracefully, not crash, when an optional dependency
is unavailable.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from .ingest import Chunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class VectorStore(ABC):
    backend_name: str = "base"

    @abstractmethod
    def build(self, chunks: List[Chunk]) -> None:
        ...

    @abstractmethod
    def search(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        ...

    def __len__(self) -> int:  # pragma: no cover - convenience only
        return getattr(self, "_size", 0)


class TfidfVectorStore(VectorStore):
    """Zero-dependency-download fallback retriever using TF-IDF + cosine similarity."""

    backend_name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = None
        self._chunks: List[Chunk] = []
        self._size = 0

    def build(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build a vector store from an empty corpus")
        self._chunks = chunks
        self._matrix = self._vectorizer.fit_transform([c.text for c in chunks])
        self._size = len(chunks)

    def search(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        if self._matrix is None:
            raise RuntimeError("Vector store has not been built yet")
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        top_idx = np.argsort(-scores)[:k]
        return [RetrievedChunk(self._chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0]


class DenseVectorStore(VectorStore):
    """Semantic retriever: sentence-transformer embeddings indexed in FAISS."""

    backend_name = "dense-faiss"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        self._faiss = faiss
        self._model = SentenceTransformer(model_name)
        self._index = None
        self._chunks: List[Chunk] = []
        self._size = 0

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return vectors / norms

    def build(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build a vector store from an empty corpus")
        self._chunks = chunks
        embeddings = self._model.encode([c.text for c in chunks], convert_to_numpy=True)
        embeddings = self._normalize(embeddings.astype("float32"))
        index = self._faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self._index = index
        self._size = len(chunks)

    def search(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        if self._index is None:
            raise RuntimeError("Vector store has not been built yet")
        query_vec = self._model.encode([query], convert_to_numpy=True).astype("float32")
        query_vec = self._normalize(query_vec)
        scores, indices = self._index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(RetrievedChunk(self._chunks[idx], float(score)))
        return results


def get_vector_store(backend: str = "auto") -> VectorStore:
    """Factory: returns a DenseVectorStore when possible, else TfidfVectorStore.

    backend: "auto" (try dense, fall back to tfidf), "dense", or "tfidf".
    """
    if backend == "tfidf":
        return TfidfVectorStore()

    if backend in ("auto", "dense"):
        try:
            return DenseVectorStore()
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any init failure -> fallback
            if backend == "dense":
                raise
            logger.warning(
                "Dense embedding backend unavailable (%s); falling back to TF-IDF retrieval.",
                exc,
            )
            return TfidfVectorStore()

    raise ValueError(f"Unknown vector backend: {backend}")
