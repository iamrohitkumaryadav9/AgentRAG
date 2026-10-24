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


def test_concurrent_runs_collect_only_their_own_spans():
    """Two requests overlap inside the synthesizer; each must still report exactly its own four agent spans."""
    import threading

    barrier = threading.Barrier(2, timeout=30)

    class MeetingLLM(ExtractiveLLM):
        def generate(self, query, context_chunks, feedback=None):
            barrier.wait()  # neither run can finish its synthesizer until both are inside theirs
            return super().generate(query, context_chunks, feedback)

    store = _built_store()
    results, errors = {}, []

    def run(name, question):
        try:
            results[name] = run_query(store, question, llm=MeetingLLM())
        except Exception as exc:  # noqa: BLE001 - surfaced by the assertions below
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=("a", "What is retrieval-augmented generation?")),
        threading.Thread(target=run, args=("b", "How do guardrails work in production AI systems?")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors and set(results) == {"a", "b"}
    for name in ("a", "b"):
        spans = results[name]["metrics"].spans
        assert [s.name for s in spans] == ["router", "retriever", "synthesizer", "critic"], name
        assert results[name]["metrics"].total_ms > 0
    # the two runs retrieved different things, so their retriever spans must differ in what they recorded
    a, b = (results[n]["metrics"].spans[1].attributes for n in ("a", "b"))
    assert a["agentrag.top_score"] != b["agentrag.top_score"]


def test_metrics_do_not_leak_out_of_the_block_or_across_blocks():
    from agentrag.observability import _current_metrics

    with collect_metrics() as outer:
        with collect_metrics() as inner:
            with traced_node("inner-only"):
                pass
        with traced_node("outer-only"):
            pass
    assert [s.name for s in inner.spans] == ["inner-only"]
    assert [s.name for s in outer.spans] == ["outer-only"]
    assert _current_metrics.get() is None
