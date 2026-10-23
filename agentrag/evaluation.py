"""Offline evaluation harness: AI quality metrics for the RAG pipeline.

Running an LLM system without an eval harness means every change is a guess.
This module scores the pipeline against a fixed golden set and reports the
metrics that actually decide whether a RAG system is fit for production:

    retrieval_hit_rate    did the expected source document appear in top-k?
                          (isolates retrieval failure from generation failure)
    groundedness_rate     what share of answers cleared the guardrail?
    keyword_recall        did the answer contain the facts we expect?
    refusal_accuracy      on deliberately out-of-scope questions, did the
                          system correctly refuse instead of confabulating?
    citation_validity     were all cited sources real (not fabricated)?
    p50 / p95 latency     tail latency, not just the average
    est_tokens_per_query  cost proxy, tracked so regressions are visible

The out-of-scope cases matter as much as the in-scope ones: a system that
scores well on questions it can answer but hallucinates on questions it
can't is not safe to deploy, and an accuracy-only metric hides that
completely.

Thresholds in ``DEFAULT_THRESHOLDS`` are enforced by CI, so a change that
degrades answer quality fails the build the same way a broken unit test
would.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from .graph import run_query
from .llm import LLM
from .vectorstore import VectorStore


@dataclass
class EvalCase:
    """One golden-set question.

    expected_source: the document that should be retrieved (None => out of scope)
    expected_keywords: terms a correct answer should mention
    should_refuse: True for out-of-scope questions the system must decline
    """

    question: str
    expected_source: str | None = None
    expected_keywords: List[str] = field(default_factory=list)
    should_refuse: bool = False


GOLDEN_SET: List[EvalCase] = [
    # ---- in-scope: retrieval + grounded generation ----
    EvalCase(
        question="What is retrieval-augmented generation?",
        expected_source="rag_systems.md",
        expected_keywords=["retrieval", "language model"],
    ),
    EvalCase(
        question="Why does chunk size matter in a RAG pipeline?",
        expected_source="rag_systems.md",
        expected_keywords=["chunk"],
    ),
    EvalCase(
        question="What are the main risks of a RAG system?",
        expected_source="rag_systems.md",
        expected_keywords=["retrieval", "chunking"],
    ),
    EvalCase(
        question="Which vector stores are commonly used for RAG?",
        expected_source="rag_systems.md",
        expected_keywords=["vector"],
    ),
    EvalCase(
        question="What is the router worker critic pattern?",
        expected_source="agentic_ai.md",
        expected_keywords=["router", "critic"],
    ),
    EvalCase(
        question="How does LangGraph model a multi-agent workflow?",
        expected_source="agentic_ai.md",
        expected_keywords=["graph", "node"],
    ),
    EvalCase(
        question="Why should an agent retry loop be bounded?",
        expected_source="agentic_ai.md",
        expected_keywords=["loop", "attempts"],
    ),
    EvalCase(
        question="What guardrails should production AI systems have?",
        expected_source="responsible_ai.md",
        expected_keywords=["guardrail", "groundedness"],
    ),
    EvalCase(
        question="How is observability handled for AI systems?",
        expected_source="responsible_ai.md",
        expected_keywords=["latency", "cost"],
    ),
    EvalCase(
        question="What makes an AI system explainable and auditable?",
        expected_source="responsible_ai.md",
        expected_keywords=["auditable", "source"],
    ),
    # ---- out of scope: the system must refuse, not confabulate ----
    EvalCase(question="What was Tesla's share price yesterday?", should_refuse=True),
    EvalCase(question="Who won the 2026 FIFA World Cup final?", should_refuse=True),
]


DEFAULT_THRESHOLDS: Dict[str, float] = {
    "retrieval_hit_rate": 0.80,
    "groundedness_rate": 0.90,
    "keyword_recall": 0.70,
    "refusal_accuracy": 1.00,
    "citation_validity": 1.00,
}


@dataclass
class CaseResult:
    question: str
    passed_grounding: bool
    retrieval_hit: bool | None
    keyword_hit: bool | None
    refused_correctly: bool | None
    citations_valid: bool
    latency_ms: float
    est_tokens: int
    answer: str


def _answer_is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(
        phrase in lowered
        for phrase in ("don't have enough information", "not enough information", "cannot answer", "no information")
    )


def evaluate(vector_store: VectorStore, llm: LLM | None = None, cases: List[EvalCase] | None = None) -> Dict[str, Any]:
    """Run the golden set through the pipeline and compute quality metrics."""
    cases = cases or GOLDEN_SET
    results: List[CaseResult] = []

    for case in cases:
        start = time.perf_counter()
        state = run_query(vector_store, case.question, llm=llm)
        latency_ms = (time.perf_counter() - start) * 1000

        answer = state.get("answer", "")
        guardrail = state.get("guardrail")
        retrieved = state.get("retrieved", [])
        metrics = state.get("metrics")

        retrieval_hit: bool | None = None
        keyword_hit: bool | None = None
        refused_correctly: bool | None = None

        if case.should_refuse:
            # Correct behaviour is an explicit refusal, or a guardrail block --
            # both mean the system declined to assert something unsupported.
            refused_correctly = _answer_is_refusal(answer) or (guardrail is not None and not guardrail.passed)
        else:
            sources = {r.chunk.source for r in retrieved}
            retrieval_hit = case.expected_source in sources
            lowered = answer.lower()
            keyword_hit = all(kw.lower() in lowered for kw in case.expected_keywords)

        results.append(
            CaseResult(
                question=case.question,
                passed_grounding=bool(guardrail.passed) if guardrail else True,
                retrieval_hit=retrieval_hit,
                keyword_hit=keyword_hit,
                refused_correctly=refused_correctly,
                citations_valid=not state.get("bad_citations"),
                latency_ms=round(latency_ms, 2),
                est_tokens=metrics.est_tokens if metrics else 0,
                answer=answer,
            )
        )

    return _summarise(results)


def _rate(values: List[bool]) -> float:
    return round(sum(1 for v in values if v) / len(values), 3) if values else 1.0


def _summarise(results: List[CaseResult]) -> Dict[str, Any]:
    latencies = sorted(r.latency_ms for r in results)
    in_scope = [r for r in results if r.retrieval_hit is not None]
    out_scope = [r for r in results if r.refused_correctly is not None]

    def percentile(p: float) -> float:
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, int(round(p * (len(latencies) - 1))))
        return round(latencies[idx], 2)

    metrics = {
        "cases": len(results),
        "retrieval_hit_rate": _rate([bool(r.retrieval_hit) for r in in_scope]),
        "groundedness_rate": _rate([r.passed_grounding for r in results]),
        "keyword_recall": _rate([bool(r.keyword_hit) for r in in_scope]),
        "refusal_accuracy": _rate([bool(r.refused_correctly) for r in out_scope]),
        "citation_validity": _rate([r.citations_valid for r in results]),
        "latency_p50_ms": percentile(0.50),
        "latency_p95_ms": percentile(0.95),
        "mean_est_tokens_per_query": int(statistics.mean([r.est_tokens for r in results])) if results else 0,
    }

    return {"metrics": metrics, "results": [asdict(r) for r in results]}


def check_thresholds(metrics: Dict[str, Any], thresholds: Dict[str, float] | None = None) -> List[str]:
    """Return a list of threshold violations (empty means the quality gate passed)."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    failures = []
    for name, minimum in thresholds.items():
        actual = metrics.get(name)
        if actual is not None and actual < minimum:
            failures.append(f"{name}={actual:.3f} below required {minimum:.2f}")
    return failures


def format_report(report: Dict[str, Any]) -> str:
    """Render the metrics as a markdown table (used in CI job summaries)."""
    m = report["metrics"]
    lines = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Cases | {m['cases']} |",
        f"| Retrieval hit-rate@k | {m['retrieval_hit_rate']:.1%} |",
        f"| Groundedness pass rate | {m['groundedness_rate']:.1%} |",
        f"| Keyword recall | {m['keyword_recall']:.1%} |",
        f"| Refusal accuracy (out-of-scope) | {m['refusal_accuracy']:.1%} |",
        f"| Citation validity | {m['citation_validity']:.1%} |",
        f"| Latency p50 | {m['latency_p50_ms']} ms |",
        f"| Latency p95 | {m['latency_p95_ms']} ms |",
        f"| Mean est. tokens/query | {m['mean_est_tokens_per_query']} |",
    ]
    return "\n".join(lines)


def to_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=2)
