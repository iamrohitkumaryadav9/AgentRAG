from agentrag.evaluation import DEFAULT_THRESHOLDS, EvalCase, check_thresholds, evaluate, format_report
from agentrag.ingest import build_corpus
from agentrag.llm import ExtractiveLLM
from agentrag.vectorstore import TfidfVectorStore


def _built_store():
    store = TfidfVectorStore()
    store.build(build_corpus("data/sample_docs", chunk_size=500, chunk_overlap=80))
    return store


def test_evaluation_meets_quality_thresholds():
    """The golden-set eval is the quality gate CI enforces."""
    report = evaluate(_built_store(), llm=ExtractiveLLM())
    failures = check_thresholds(report["metrics"], DEFAULT_THRESHOLDS)
    assert not failures, f"quality gate violations: {failures}"


def test_evaluation_reports_all_metrics():
    report = evaluate(_built_store(), llm=ExtractiveLLM())
    metrics = report["metrics"]
    for key in (
        "retrieval_hit_rate",
        "groundedness_rate",
        "keyword_recall",
        "refusal_accuracy",
        "citation_validity",
        "latency_p50_ms",
        "latency_p95_ms",
        "mean_est_tokens_per_query",
    ):
        assert key in metrics, f"missing metric: {key}"
    assert metrics["cases"] > 0


def test_out_of_scope_question_is_refused_not_confabulated():
    cases = [EvalCase(question="What is the capital of Uzbekistan?", should_refuse=True)]
    report = evaluate(_built_store(), llm=ExtractiveLLM(), cases=cases)
    assert report["metrics"]["refusal_accuracy"] == 1.0


def test_citations_are_never_fabricated():
    report = evaluate(_built_store(), llm=ExtractiveLLM())
    assert report["metrics"]["citation_validity"] == 1.0


def test_format_report_renders_markdown_table():
    report = evaluate(_built_store(), llm=ExtractiveLLM())
    markdown = format_report(report)
    assert "| Metric | Value |" in markdown
    assert "Retrieval hit-rate@k" in markdown


def test_refusal_behaviour_is_stable_across_chunk_configs():
    """The relevance gate must not be tuned to one chunking configuration.

    A raw similarity floor passed at chunk_size=800 but let an out-of-scope
    question through at chunk_size=500. Query-term coverage is normalised, so
    refusal accuracy holds under both.
    """
    for chunk_size, overlap in [(500, 80), (800, 120), (1200, 150)]:
        store = TfidfVectorStore()
        store.build(build_corpus("data/sample_docs", chunk_size=chunk_size, chunk_overlap=overlap))
        report = evaluate(store, llm=ExtractiveLLM())
        assert report["metrics"]["refusal_accuracy"] == 1.0, (
            f"refusal accuracy regressed at chunk_size={chunk_size}"
        )
