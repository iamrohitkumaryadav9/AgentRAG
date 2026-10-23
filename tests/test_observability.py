from agentrag.graph import run_query
from agentrag.ingest import build_corpus
from agentrag.llm import ExtractiveLLM
from agentrag.observability import collect_metrics, estimate_tokens, traced_node
from agentrag.vectorstore import TfidfVectorStore


def _built_store():
    store = TfidfVectorStore()
    store.build(build_corpus("data/sample_docs", chunk_size=500, chunk_overlap=80))
    return store


def test_traced_node_records_span_with_attributes():
    with collect_metrics() as metrics:
        with traced_node("demo", backend="test") as span:
            span["custom_value"] = 42

    assert len(metrics.spans) == 1
    span_record = metrics.spans[0]
    assert span_record.name == "demo"
    assert span_record.duration_ms >= 0
    assert span_record.attributes["agentrag.custom_value"] == 42
    assert span_record.attributes["agentrag.backend"] == "test"


def test_query_run_emits_span_per_agent_node():
    result = run_query(_built_store(), "What is retrieval-augmented generation?", llm=ExtractiveLLM())
    metrics = result["metrics"]

    node_names = [s.name for s in metrics.spans]
    for expected in ("router", "retriever", "synthesizer", "critic"):
        assert expected in node_names, f"no span emitted for {expected}"

    assert metrics.total_ms > 0
    assert metrics.est_tokens > 0
    assert metrics.node_ms("retriever") >= 0


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)
