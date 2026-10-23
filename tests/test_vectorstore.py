import pytest

import agentrag.vectorstore as vs
from agentrag.fusion import rrf_fuse
from agentrag.ingest import Chunk, build_corpus
from agentrag.vectorstore import (
    Bm25VectorStore,
    HybridVectorStore,
    TfidfVectorStore,
    VectorStore,
    get_vector_store,
)


def _corpus():
    return build_corpus("data/sample_docs", chunk_size=500, chunk_overlap=80)


def _chunks(*texts):
    return [Chunk(id=f"c{i}", source=f"d{i}.md", text=t, position=0) for i, t in enumerate(texts)]


def test_tfidf_store_retrieves_relevant_chunk():
    store = TfidfVectorStore()
    store.build(_corpus())

    results = store.search("What is retrieval-augmented generation?", k=3)
    assert results
    # the RAG explainer doc should rank first for a RAG-specific query
    assert results[0].chunk.source == "rag_systems.md"
    assert results[0].score > 0


def test_tfidf_store_empty_corpus_raises():
    store = TfidfVectorStore()
    with pytest.raises(ValueError):
        store.build([])


def test_get_vector_store_auto_never_crashes():
    # "auto" must always return a usable store, even if the dense backend
    # (which needs a network download) is unavailable in this environment.
    store = get_vector_store("auto")
    store.build(_corpus())
    results = store.search("multi-agent system", k=2)
    assert results


# ---- BM25 -------------------------------------------------------------------


def test_bm25_ranks_exact_rare_term_first():
    store = Bm25VectorStore()
    store.build(_chunks(
        "general discussion of cells and tissue growth",
        "the MDA5 receptor senses viral RNA in the cytoplasm",
        "tissue engineering with scaffolds",
    ))
    results = store.search("MDA5", k=3)
    assert [r.chunk.id for r in results] == ["c1"]  # only the matching chunk is returned


def test_bm25_returns_nothing_when_no_query_term_matches():
    store = Bm25VectorStore()
    store.build(_chunks("alpha beta gamma", "delta epsilon"))
    assert store.search("zeta", k=3) == []
    assert store.search("the and of", k=3) == []  # stop words only


def test_bm25_term_in_every_document_still_scores_positive():
    """Lucene-style idf never goes negative, unlike the textbook formula."""
    store = Bm25VectorStore()
    store.build(_chunks("common word here", "common word there", "common word everywhere"))
    assert all(r.score > 0 for r in store.search("common", k=3))


def test_bm25_prefers_shorter_document_for_same_term_count():
    store = Bm25VectorStore()
    store.build(_chunks("retrieval " + "filler " * 40, "retrieval filler"))
    assert store.search("retrieval", k=2)[0].chunk.id == "c1"


def test_bm25_empty_corpus_and_unbuilt_store_raise():
    with pytest.raises(ValueError):
        Bm25VectorStore().build([])
    with pytest.raises(RuntimeError):
        Bm25VectorStore().search("anything")


def test_bm25_retrieves_relevant_sample_doc():
    store = Bm25VectorStore()
    store.build(_corpus())
    assert store.search("What is retrieval-augmented generation?", k=3)[0].chunk.source == "rag_systems.md"


# ---- fusion / hybrid ----------------------------------------------------------


def test_rrf_rewards_documents_ranked_well_by_both_lists():
    # "b" is 2nd in both lists (1/62 + 1/62 = 0.03226); "a" is 1st in one and 5th in the other
    # (1/61 + 1/65 = 0.03178), so agreement wins.
    fused = rrf_fuse([["a", "b", "c", "d", "e"], ["z", "b", "y", "x", "a"]], k=60, limit=3)
    assert [doc for doc, _ in fused][:2] == ["b", "a"]


def test_hybrid_score_is_normalised_and_one_means_unanimous_first():
    corpus = _chunks("rare token zebra stripes", "unrelated filler text", "more unrelated filler")
    hybrid = HybridVectorStore(parts=[Bm25VectorStore(), TfidfVectorStore()])
    hybrid.build(corpus)
    results = hybrid.search("zebra", k=3)
    assert results[0].chunk.id == "c0"
    assert results[0].score == pytest.approx(1.0)
    assert all(0 < r.score <= 1.0 for r in results)


def test_hybrid_surfaces_a_chunk_only_one_part_finds():
    class Silent(VectorStore):
        backend_name = "silent"

        def build(self, chunks):
            pass

        def search(self, query, k=4):
            return []

    corpus = _chunks("rare token zebra", "something else entirely")
    hybrid = HybridVectorStore(parts=[Bm25VectorStore(), Silent()])
    hybrid.build(corpus)
    top = hybrid.search("zebra", k=2)[0]
    assert top.chunk.id == "c0"
    assert top.score == pytest.approx(0.5)  # found by one of two retrievers


# ---- backend selection ----------------------------------------------------------


def test_auto_falls_back_to_bm25_not_tfidf_when_dense_is_unavailable(monkeypatch):
    class Broken:
        def __init__(self, *args, **kwargs):
            raise ImportError("No module named 'faiss'")

    monkeypatch.setattr(vs, "DenseVectorStore", Broken)
    store = get_vector_store("auto")
    assert isinstance(store, Bm25VectorStore)
    assert store.backend_name == "bm25"


def test_auto_picks_hybrid_when_dense_is_available(monkeypatch):
    class FakeDense(VectorStore):
        backend_name = "fake-dense"

        def __init__(self, model_name="", query_prefix=""):
            self.model_name, self.query_prefix = model_name, query_prefix

        def build(self, chunks):
            pass

        def search(self, query, k=4):
            return []

    monkeypatch.setattr(vs, "DenseVectorStore", FakeDense)
    store = get_vector_store("auto")
    assert isinstance(store, HybridVectorStore)
    dense = store.parts[1]
    assert dense.model_name == "BAAI/bge-small-en-v1.5"
    assert dense.query_prefix.startswith("Represent this sentence")


def test_explicit_backends_do_not_fall_back(monkeypatch):
    class Broken:
        def __init__(self, *args, **kwargs):
            raise ImportError("missing")

    monkeypatch.setattr(vs, "DenseVectorStore", Broken)
    with pytest.raises(ImportError):
        get_vector_store("dense")
    with pytest.raises(ImportError):
        get_vector_store("hybrid")
    assert get_vector_store("bm25").backend_name == "bm25"
    assert get_vector_store("tfidf").backend_name == "tfidf"


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_vector_store("nonsense")
