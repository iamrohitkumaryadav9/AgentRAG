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

_TOKEN_RE = re.compile(r"\w+")


class Retriever(ABC):
    name: str = "base"

    @abstractmethod
    def index(self, corpus: dict[str, str]) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]: ...


def _top_k(scores: np.ndarray, doc_ids: list[str], k: int) -> list[tuple[str, float]]:
    k = min(k, len(doc_ids))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    return [(doc_ids[i], float(scores[i])) for i in top]


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

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        vec = self._load_model().encode(
            [self.query_prefix + query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, idx = self._index.search(vec, min(k, len(self._doc_ids)))
        return [(self._doc_ids[i], float(s)) for s, i in zip(scores[0], idx[0], strict=True) if i != -1]
