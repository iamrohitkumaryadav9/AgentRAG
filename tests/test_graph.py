from agentrag.graph import run_query
from agentrag.ingest import build_corpus
from agentrag.llm import ExtractiveLLM
from agentrag.vectorstore import TfidfVectorStore


def _built_store():
    store = TfidfVectorStore()
    store.build(build_corpus("data/sample_docs", chunk_size=500, chunk_overlap=80))
    return store


def test_end_to_end_query_is_grounded():
    store = _built_store()
    result = run_query(store, "What is retrieval-augmented generation?", llm=ExtractiveLLM())

    assert result["answer"]
    assert result["retrieved"]
    assert result["guardrail"].passed
    assert result["attempts"] >= 1
    assert any("router" in step for step in result["trace"])
    assert any("critic" in step for step in result["trace"])


def test_small_talk_routes_directly_without_retrieval():
    store = _built_store()
    result = run_query(store, "hello", llm=ExtractiveLLM())

    assert result["route"] == "direct"
    assert "retrieved" not in result or result["retrieved"] == []


def test_query_about_guardrails_is_grounded_in_responsible_ai_doc():
    store = _built_store()
    result = run_query(store, "How do guardrails work in production AI systems?", llm=ExtractiveLLM())

    sources = {r.chunk.source for r in result["retrieved"]}
    assert "responsible_ai.md" in sources
    assert result["guardrail"].passed


def test_out_of_scope_query_is_refused_via_relevance_floor():
    """Retrieval below the relevance floor must abstain, not synthesise."""
    store = _built_store()
    result = run_query(store, "What was Tesla's share price yesterday?", llm=ExtractiveLLM())

    assert "don't have enough information" in result["answer"].lower()
    assert result["payload"].confidence == 0.0
    assert result["payload"].citations == []
    assert any("refuse" in step for step in result["trace"])


def test_in_scope_query_still_reaches_synthesizer():
    store = _built_store()
    result = run_query(store, "What is retrieval-augmented generation?", llm=ExtractiveLLM())

    assert any("synthesizer" in step for step in result["trace"])
    assert result["payload"].citations
