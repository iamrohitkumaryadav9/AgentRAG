"""Retrieval systems under test, behind one tiny interface.

Each system indexes ``{doc_id: text}`` and answers ``search(query, k)`` with
``[(doc_id, score)]`` best-first. Heavy dependencies are imported lazily so
importing this module is cheap.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from agentrag.fusion import rrf_fuse

_TOKEN_RE = re.compile(r"\w+")


class Retriever(ABC):
    name: str = "base"
    indexed: bool = False

    @abstractmethod
    def index(self, corpus: dict[str, str]) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]: ...

    def search_batch(self, queries: list[str], k: int = 100) -> list[list[tuple[str, float]]]:
        """Answer many queries at once. Subclasses override this when batching is cheaper."""
        return [self.search(q, k=k) for q in queries]


def _top_k(scores: np.ndarray, doc_ids: list[str], k: int) -> list[tuple[str, float]]:
    k = min(k, len(doc_ids))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    return [(doc_ids[i], float(scores[i])) for i in top]


class StoreRetriever(Retriever):
    """Adapter that runs one of the app's own ``VectorStore`` backends through the benchmark.

    Benchmarking the shipped classes (not a re-implementation) is what lets the
    results speak for the app's defaults.
    """

    def __init__(self, name: str, store_factory) -> None:
        self.name = name
        self._store = store_factory()

    def index(self, corpus: dict[str, str]) -> None:
        from agentrag.ingest import Chunk

        self._store.build([Chunk(id=d, source=d, text=t, position=0) for d, t in corpus.items()])
        self.indexed = True

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        return [(r.chunk.source, r.score) for r in self._store.search(query, k=k)]


class TfidfRetriever(StoreRetriever):
    """The app's TfidfVectorStore (1-2 grams, English stopwords), as a baseline."""

    def __init__(self) -> None:
        from agentrag.vectorstore import TfidfVectorStore

        super().__init__("tfidf", TfidfVectorStore)


class BM25Retriever(Retriever):
    """Okapi BM25 over lowercased word tokens with English stopwords removed."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._bm25 = None
        self._doc_ids: list[str] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        return [t for t in _TOKEN_RE.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]

    def index(self, corpus: dict[str, str]) -> None:
        from rank_bm25 import BM25Okapi

        self._doc_ids = list(corpus)
        self._bm25 = BM25Okapi([self.tokenize(corpus[d]) for d in self._doc_ids], k1=self.k1, b=self.b)
        self.indexed = True

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        scores = np.asarray(self._bm25.get_scores(self.tokenize(query)))
        return _top_k(scores, self._doc_ids, k)


class DenseRetriever(Retriever):
    """Sentence-transformer embeddings with exact (flat inner-product) FAISS search.

    Vectors are L2-normalised, so inner product equals cosine similarity.
    Document embeddings are cached on disk, keyed by model name.
    """

    def __init__(self, model_name: str, query_prefix: str = "", cache_dir: Path | None = None) -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.name = f"dense:{model_name.split('/')[-1]}"
        self._cache_dir = cache_dir
        self._model = None
        self._index = None
        self._doc_ids: list[str] = []

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def index(self, corpus: dict[str, str]) -> None:
        import faiss

        self._doc_ids = list(corpus)
        cache_file = None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self._cache_dir / f"{self.model_name.replace('/', '__')}-{len(corpus)}.npy"

        if cache_file is not None and cache_file.exists():
            embeddings = np.load(cache_file)
        else:
            embeddings = self._load_model().encode(
                [corpus[d] for d in self._doc_ids],
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            ).astype("float32")
            if cache_file is not None:
                np.save(cache_file, embeddings)

        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)
        self.indexed = True

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        vec = self._load_model().encode(
            [self.query_prefix + query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, idx = self._index.search(vec, min(k, len(self._doc_ids)))
        return [(self._doc_ids[i], float(s)) for s, i in zip(scores[0], idx[0], strict=True) if i != -1]

    def search_batch(self, queries: list[str], k: int = 100) -> list[list[tuple[str, float]]]:
        """Encode all queries in batches and run one FAISS search for the whole matrix."""
        vecs = self._load_model().encode(
            [self.query_prefix + q for q in queries],
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        scores, idx = self._index.search(vecs, min(k, len(self._doc_ids)))
        return [
            [(self._doc_ids[i], float(s)) for s, i in zip(row_s, row_i, strict=True) if i != -1]
            for row_s, row_i in zip(scores, idx, strict=True)
        ]


class HybridRRFRetriever(Retriever):
    """Fuse several retrievers' top-``depth`` lists with reciprocal rank fusion."""

    def __init__(self, parts: list[Retriever], k_rrf: int = 60, depth: int = 100) -> None:
        self.parts, self.k_rrf, self.depth = parts, k_rrf, depth
        self.name = "hybrid-rrf(" + "+".join(p.name for p in parts) + ")"

    def index(self, corpus: dict[str, str]) -> None:
        for part in self.parts:
            if not part.indexed:
                part.index(corpus)
        self.indexed = True

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        rankings = [[d for d, _ in part.search(query, k=self.depth)] for part in self.parts]
        return rrf_fuse(rankings, k=self.k_rrf, limit=k)

    def search_batch(self, queries: list[str], k: int = 100) -> list[list[tuple[str, float]]]:
        per_part = [part.search_batch(queries, k=self.depth) for part in self.parts]
        return [
            rrf_fuse([[d for d, _ in results[i]] for results in per_part], k=self.k_rrf, limit=k)
            for i in range(len(queries))
        ]


class CrossEncoderReranker(Retriever):
    """Rerank the head of a first-stage ranking with a cross-encoder.

    Only the top ``rerank_depth`` candidates are rescored; the rest keep their
    first-stage order below them. That makes the reranker's ceiling explicit:
    Recall@``rerank_depth`` can never exceed the first stage's, because
    reranking only permutes the candidate pool.
    """

    def __init__(
        self, first_stage: Retriever, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", rerank_depth: int = 50
    ) -> None:
        self.first_stage, self.model_name, self.rerank_depth = first_stage, model_name, rerank_depth
        self.name = f"{first_stage.name}+rerank({model_name.split('/')[-1]}@{rerank_depth})"
        self._model = None
        self._corpus: dict[str, str] = {}

    def index(self, corpus: dict[str, str]) -> None:
        if not self.first_stage.indexed:
            self.first_stage.index(corpus)
        self._corpus = corpus
        self.indexed = True

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def _rerank(self, query: str, candidates: list[tuple[str, float]], scores, k: int) -> list[tuple[str, float]]:
        head, tail = candidates[: self.rerank_depth], candidates[self.rerank_depth :]
        reranked = sorted(zip((d for d, _ in head), scores, strict=True), key=lambda item: -item[1])
        return ([(d, float(s)) for d, s in reranked] + tail)[:k]

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        model = self._load_model()
        candidates = self.first_stage.search(query, k=max(k, self.rerank_depth))
        pairs = [(query, self._corpus[d]) for d, _ in candidates[: self.rerank_depth]]
        scores = model.predict(pairs, batch_size=25, show_progress_bar=False)
        return self._rerank(query, candidates, scores, k)

    def search_batch(self, queries: list[str], k: int = 100) -> list[list[tuple[str, float]]]:
        """Batch the first stage, then score every (query, candidate) pair in one cross-encoder pass."""
        model = self._load_model()
        all_candidates = self.first_stage.search_batch(queries, k=max(k, self.rerank_depth))
        pairs = [
            (q, self._corpus[d]) for q, cands in zip(queries, all_candidates, strict=True)
            for d, _ in cands[: self.rerank_depth]
        ]
        flat = model.predict(pairs, batch_size=32, show_progress_bar=False)
        out, offset = [], 0
        for q, cands in zip(queries, all_candidates, strict=True):
            n = min(len(cands), self.rerank_depth)
            out.append(self._rerank(q, cands, flat[offset : offset + n], k))
            offset += n
        return out
