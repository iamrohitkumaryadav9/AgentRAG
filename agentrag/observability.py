"""OpenTelemetry instrumentation for the agent graph.

Every agent node emits a span carrying the attributes you actually need to
debug and cost-manage a production agentic system:

    agentrag.node             which agent ran
    agentrag.retrieved_count  how many chunks came back
    agentrag.top_score        best retrieval score (retrieval quality signal)
    agentrag.attempt          which synthesis attempt this is (retry depth)
    agentrag.grounding_score  the critic's groundedness score
    agentrag.guardrail_passed whether the answer cleared the guardrail
    agentrag.llm_backend      which generation backend served the request
    agentrag.est_tokens       rough token estimate (cost signal)

By default spans are collected in-process and summarised per request, so the
CLI/API can report a per-query trace without needing a collector running.
Set ``AGENTRAG_OTEL_CONSOLE=1`` to also print raw spans to stdout, or point
``OTEL_EXPORTER_OTLP_ENDPOINT`` at a collector and set
``AGENTRAG_OTEL_OTLP=1`` to ship them to a real observability backend
(Jaeger, Tempo, Azure Monitor, ...) without any code change.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

_TRACER_NAME = "agentrag"
_initialised = False


def _init_tracing() -> None:
    global _initialised
    if _initialised:
        return

    provider = TracerProvider(resource=Resource.create({"service.name": "agentrag"}))

    if os.getenv("AGENTRAG_OTEL_CONSOLE") == "1":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    if os.getenv("AGENTRAG_OTEL_OTLP") == "1":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        except ImportError:
            # Optional exporter not installed -- tracing still works locally.
            pass

    trace.set_tracer_provider(provider)
    _initialised = True


def get_tracer():
    _init_tracing()
    return trace.get_tracer(_TRACER_NAME)


# --------------------------------------------------------------------------
# In-process span collection
#
# The OTel SDK is the transport; this collector is what lets a single request
# report its own timing breakdown back to the caller (CLI output, API
# response, eval harness) without requiring an external backend.
# --------------------------------------------------------------------------


@dataclass
class SpanRecord:
    name: str
    duration_ms: float
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunMetrics:
    """Per-request telemetry, assembled from the spans of one graph run."""

    spans: List[SpanRecord] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return round(sum(s.duration_ms for s in self.spans), 2)

    def node_ms(self, node: str) -> float:
        return round(sum(s.duration_ms for s in self.spans if s.name == node), 2)

    @property
    def est_tokens(self) -> int:
        return int(sum(s.attributes.get("agentrag.est_tokens", 0) for s in self.spans))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_ms": self.total_ms,
            "est_tokens": self.est_tokens,
            "spans": [
                {"name": s.name, "duration_ms": s.duration_ms, **s.attributes} for s in self.spans
            ],
        }


_current_metrics: RunMetrics | None = None


@contextmanager
def collect_metrics() -> Iterator[RunMetrics]:
    """Collect all spans emitted inside this block into one RunMetrics."""
    global _current_metrics
    previous = _current_metrics
    metrics = RunMetrics()
    _current_metrics = metrics
    try:
        yield metrics
    finally:
        _current_metrics = previous


@contextmanager
def traced_node(name: str, **attributes: Any) -> Iterator[Dict[str, Any]]:
    """Trace one agent node.

    Yields a mutable dict; anything the node writes into it is attached to
    the span as an attribute (so a node can record values it only computes
    partway through, e.g. the grounding score).
    """
    tracer = get_tracer()
    extra: Dict[str, Any] = {}
    start = time.perf_counter()

    with tracer.start_as_current_span(f"agentrag.{name}") as span:
        span.set_attribute("agentrag.node", name)
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(f"agentrag.{key}", value)
        try:
            yield extra
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            for key, value in extra.items():
                if value is not None:
                    span.set_attribute(f"agentrag.{key}", value)
            span.set_attribute("agentrag.duration_ms", round(duration_ms, 2))

            if _current_metrics is not None:
                merged = {f"agentrag.{k}": v for k, v in {**attributes, **extra}.items() if v is not None}
                _current_metrics.spans.append(
                    SpanRecord(name=name, duration_ms=round(duration_ms, 2), attributes=merged)
                )


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used as a cost proxy.

    Deliberately provider-agnostic: the point is to make cost visible and
    trackable over time, not to bill against it.
    """
    return max(1, len(text) // 4)
