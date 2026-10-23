from agentrag.ingest import build_corpus
from agentrag.vectorstore import TfidfVectorStore, get_vector_store


def _corpus():
    return build_corpus("data/sample_docs", chunk_size=500, chunk_overlap=80)


def test_tfidf_store_retrieves_relevant_chunk():
    store = TfidfVectorStore()
    store.build(_corpus())

    results = store.search("What is retrieval-augmented generation?", k=3)
    assert results
    # the RAG explainer doc should rank first for a RAG-specific query
    assert results[0].chunk.source == "rag_systems.md"
    assert results[0].score > 0


def test_tfidf_store_empty_corpus_raises():
    import pytest

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
