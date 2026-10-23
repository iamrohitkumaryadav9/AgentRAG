"""Pluggable retrieval backends sharing one interface.

* ``HybridVectorStore`` - BM25 + a dense retriever, fused with reciprocal rank
  fusion (RRF). The default under ``backend="auto"``. See docs/scifact_ablation.md
  for the evidence: it reaches the deepest recall of the strategies tested and
  keeps BM25's exact-term matching where a dense model misses rare identifiers.
* ``Bm25VectorStore``   - Okapi BM25 over scikit-learn/scipy/numpy only (no
  extra dependencies, no model download). The ``auto`` fallback whenever the
  dense side can't be initialised.
* ``DenseVectorStore``  - sentence-transformer embeddings + FAISS cosine search
  (needs the ``dense`` extra and model weights from Hugging Face on first run).
* ``TfidfVectorStore``  - scikit-learn TF-IDF + cosine similarity. Kept for CI
  and as an explicit choice; no longer the automatic fallback because BM25
  beats it by a wide margin on the benchmark at the same dependency cost.

``get_vector_store(backend="auto")`` picks the best available backend at
runtime -- a small but deliberate resilience pattern: a production RAG
service should degrade gracefully, not crash, when an optional dependency
is unavailable.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import List

import numpy as np

from .fusion import rrf_fuse
from .ingest import Chunk

logger = logging.getLogger(__name__)

# The dense half of the hybrid default: best single retriever in the SciFact ablation.
HYBRID_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_TOKEN_RE = re.compile(r"\w+")


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

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", query_prefix: str = "") -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        self._faiss = faiss
        self._model = SentenceTransformer(model_name)
        self._query_prefix = query_prefix
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
        query_vec = self._model.encode([self._query_prefix + query], convert_to_numpy=True).astype("float32")
        query_vec = self._normalize(query_vec)
        scores, indices = self._index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx == -1:
                continue
            results.append(RetrievedChunk(self._chunks[idx], float(score)))
        return results


class Bm25VectorStore(VectorStore):
    """Okapi BM25 (k1=1.5, b=0.75, Lucene-style idf) on scikit-learn/scipy/numpy alone.

    Lower-cased word tokens with English stop words removed. Chunks that share
    no term with the query score zero and are not returned. Per-(chunk, term)
    BM25 weights are precomputed at build time, so a query is a sparse column sum.
    """

    backend_name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._chunks: List[Chunk] = []
        self._vocab: dict[str, int] = {}
        self._weights = None
        self._size = 0

    @staticmethod
    def tokenize(text: str) -> list[str]:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        return [t for t in _TOKEN_RE.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]

    def build(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build a vector store from an empty corpus")
        from scipy import sparse

        docs = [self.tokenize(c.text) for c in chunks]
        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        counts: list[int] = []
        for i, tokens in enumerate(docs):
            for term, tf in Counter(tokens).items():
                rows.append(i)
                cols.append(vocab.setdefault(term, len(vocab)))
                counts.append(tf)

        n = len(chunks)
        doc_len = np.array([len(d) for d in docs], dtype=float)
        avg_len = doc_len.mean() or 1.0
        rows_a, cols_a, tf_a = np.array(rows), np.array(cols), np.array(counts, dtype=float)
        df = np.bincount(cols_a, minlength=len(vocab))
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        norm = self.k1 * (1.0 - self.b + self.b * doc_len[rows_a] / avg_len)
        weights = idf[cols_a] * tf_a * (self.k1 + 1.0) / (tf_a + norm)

        self._weights = sparse.csc_matrix((weights, (rows_a, cols_a)), shape=(n, len(vocab)))
        self._vocab, self._chunks, self._size = vocab, chunks, n

    def search(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        if self._weights is None:
            raise RuntimeError("Vector store has not been built yet")
        # Repeated query terms are counted repeatedly, as in standard BM25.
        term_ids = [self._vocab[t] for t in self.tokenize(query) if t in self._vocab]
        if not term_ids:
            return []
        scores = np.asarray(self._weights[:, term_ids].sum(axis=1)).ravel()
        top = np.argsort(-scores, kind="stable")[:k]
        return [RetrievedChunk(self._chunks[i], float(scores[i])) for i in top if scores[i] > 0]


class HybridVectorStore(VectorStore):
    """Reciprocal-rank fusion of several retrievers (default: BM25 + bge-small dense).

    Each part contributes its top ``depth`` chunks; a chunk's fused score is the
    sum of 1 / (k_rrf + rank). The reported ``score`` is that sum divided by its
    maximum possible value, so it lies in (0, 1] and 1.0 means "ranked first by
    every retriever". That keeps the graph's score floor meaningful: raw RRF
    sums are tiny and would silently change what ``min_retrieval_score`` means.
    """

    backend_name = "hybrid-rrf"

    def __init__(self, parts: list[VectorStore] | None = None, k_rrf: int = 60, depth: int = 100) -> None:
        # Building the default parts is what can fail (missing extra, no model
        # weights); get_vector_store("auto") relies on that to fall back.
        self.parts = parts if parts is not None else [
            Bm25VectorStore(),
            DenseVectorStore(model_name=HYBRID_DENSE_MODEL, query_prefix=BGE_QUERY_PREFIX),
        ]
        self.k_rrf, self.depth = k_rrf, depth
        self._size = 0

    def build(self, chunks: List[Chunk]) -> None:
        for part in self.parts:
            part.build(chunks)
        self._size = len(chunks)

    def search(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        by_id: dict[str, Chunk] = {}
        rankings: list[list[str]] = []
        for part in self.parts:
            hits = part.search(query, k=self.depth)
            rankings.append([h.chunk.id for h in hits])
            by_id.update({h.chunk.id: h.chunk for h in hits})
        best = len(self.parts) / (self.k_rrf + 1)
        return [RetrievedChunk(by_id[cid], score / best) for cid, score in rrf_fuse(rankings, self.k_rrf, k)]


def get_vector_store(backend: str = "auto") -> VectorStore:
    """Factory for the retrieval backend.

    backend: "auto" (hybrid when the dense extra and model are available, else
    BM25), or an explicit "hybrid", "bm25", "dense", "tfidf". Explicit choices
    never fall back: asking for "hybrid" or "dense" without the dependencies
    raises so the misconfiguration is visible.
    """
    if backend == "tfidf":
        return TfidfVectorStore()
    if backend == "bm25":
        return Bm25VectorStore()
    if backend == "dense":
        return DenseVectorStore()
    if backend == "hybrid":
        return HybridVectorStore()

    if backend == "auto":
        try:
            return HybridVectorStore()
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any init failure -> fallback
            logger.warning(
                "Hybrid backend unavailable (%s); falling back to BM25 retrieval. "
                'Install ".[dense]" to enable BM25 + dense hybrid.',
                exc,
            )
            return Bm25VectorStore()

    raise ValueError(f"Unknown vector backend: {backend}")
