"""The multi-agent workflow, built as a LangGraph state graph.

Agents (graph nodes):

    router       -> decides whether the query needs retrieval at all
                    (e.g. "hi" doesn't; "what is RAG?" does)
    retriever    -> pulls the top-k relevant chunks from the vector store
    synthesizer  -> drafts a structured, grounded answer (AnswerPayload)
    critic       -> a guardrail agent that scores the draft for groundedness
                    and verifies its citations; on failure it routes back to
                    the synthesizer with feedback (bounded by
                    max_synthesis_attempts) instead of silently returning an
                    ungrounded answer
    direct       -> answers small-talk / meta queries without retrieval

This "generate -> critique -> retry" loop is the same self-correction
pattern used in production agentic RAG systems to keep answers honest
without a human in the loop for every request.

Every node is wrapped in an OpenTelemetry span (see observability.py), so a
single request yields a per-agent latency, retrieval-quality, and cost
breakdown -- the raw material for monitoring an agent in production.
"""
from __future__ import annotations

from typing import List, TypedDict

from langgraph.graph import END, StateGraph

from .config import settings
from .guardrails import GuardrailResult, check_citations, check_grounding, check_pii_leak, query_coverage
from .ingest import Chunk
from .llm import LLM, get_llm
from .observability import RunMetrics, collect_metrics, estimate_tokens, traced_node
from .schemas import AnswerPayload
from .vectorstore import RetrievedChunk, VectorStore

SMALL_TALK_MARKERS = ("hi", "hello", "hey", "thanks", "thank you", "who are you")


class AgentState(TypedDict, total=False):
    query: str
    route: str
    retrieved: List[RetrievedChunk]
    top_score: float
    coverage: float
    payload: AnswerPayload
    answer: str
    guardrail: GuardrailResult
    pii_flags: List[str]
    bad_citations: List[str]
    attempts: int
    trace: List[str]


def build_graph(vector_store: VectorStore, llm: LLM | None = None):
    """Compile the LangGraph agent graph, bound to a given vector store / LLM."""
    llm = llm or get_llm(settings.llm_backend)

    def router_node(state: AgentState) -> AgentState:
        with traced_node("router") as span:
            query = state["query"].strip().lower()
            is_small_talk = len(query.split()) <= 3 and any(m in query for m in SMALL_TALK_MARKERS)
            route = "direct" if is_small_talk else "retrieve"
            span["route"] = route
        trace = state.get("trace", []) + [f"router -> {route}"]
        return {"route": route, "trace": trace}

    def retriever_node(state: AgentState) -> AgentState:
        with traced_node("retriever", backend=vector_store.backend_name) as span:
            results = vector_store.search(state["query"], k=settings.top_k)
            top_score = round(results[0].score, 4) if results else 0.0
            coverage = query_coverage(state["query"], [r.chunk for r in results])
            span["retrieved_count"] = len(results)
            span["top_score"] = top_score
            span["query_coverage"] = coverage
        trace = state.get("trace", []) + [
            f"retriever -> {len(results)} chunk(s), top_score={top_score}, coverage={coverage}"
        ]
        return {"retrieved": results, "top_score": top_score, "coverage": coverage, "trace": trace}

    def refuse_node(state: AgentState) -> AgentState:
        """Abstain when retrieval found nothing that actually covers the question.

        Answering from weak matches is how a RAG system ends up confidently
        wrong; refusing is the correct, auditable behaviour.
        """
        with traced_node("refuse") as span:
            top_score = state.get("top_score", 0.0)
            span["top_score"] = top_score
            span["query_coverage"] = state.get("coverage", 0.0)
            payload = AnswerPayload(
                answer=(
                    "I don't have enough information in the knowledge base to answer that. "
                    "The indexed documents don't appear to cover this topic."
                ),
                citations=[],
                confidence=0.0,
            )
        trace = state.get("trace", []) + [f"refuse -> retrieval below relevance floor (top_score={top_score})"]
        return {"payload": payload, "answer": payload.answer, "trace": trace}

    def synthesizer_node(state: AgentState) -> AgentState:
        attempts = state.get("attempts", 0) + 1
        retrieved: List[RetrievedChunk] = state.get("retrieved", [])
        chunks: List[Chunk] = [r.chunk for r in retrieved]

        feedback = None
        if state.get("guardrail") and not state["guardrail"].passed:
            feedback = state["guardrail"].reason

        with traced_node("synthesizer", llm_backend=llm.backend_name, attempt=attempts) as span:
            payload = llm.generate(state["query"], chunks, feedback=feedback)
            context_tokens = sum(estimate_tokens(c.text) for c in chunks)
            span["est_tokens"] = context_tokens + estimate_tokens(payload.answer)
            span["confidence"] = payload.confidence
            span["citation_count"] = len(payload.citations)

        trace = state.get("trace", []) + [f"synthesizer -> draft #{attempts} via {llm.backend_name}"]
        return {"payload": payload, "answer": payload.answer, "attempts": attempts, "trace": trace}

    def direct_node(state: AgentState) -> AgentState:
        with traced_node("direct"):
            payload = AnswerPayload(
                answer="Hello! Ask me a question and I'll answer it using the ingested knowledge base.",
                citations=[],
                confidence=1.0,
            )
        trace = state.get("trace", []) + ["direct -> small-talk response"]
        return {"payload": payload, "answer": payload.answer, "trace": trace}

    def critic_node(state: AgentState) -> AgentState:
        retrieved: List[RetrievedChunk] = state.get("retrieved", [])
        chunks = [r.chunk for r in retrieved]
        payload: AnswerPayload = state["payload"]

        with traced_node("critic") as span:
            result = check_grounding(payload.answer, chunks, threshold=settings.grounding_threshold)
            pii_flags = check_pii_leak(payload.answer)
            bad_citations = check_citations(payload.citations, chunks)

            # A fabricated citation is a hard failure even if the prose itself
            # happens to overlap the context.
            if bad_citations and result.passed:
                result = GuardrailResult(
                    passed=False,
                    grounding_score=result.grounding_score,
                    reason=f"cited sources not present in retrieved context: {', '.join(bad_citations)}",
                )

            span["grounding_score"] = result.grounding_score
            span["guardrail_passed"] = result.passed
            span["pii_flag_count"] = len(pii_flags)
            span["bad_citation_count"] = len(bad_citations)

        trace = state.get("trace", []) + [
            f"critic -> passed={result.passed} score={result.grounding_score} ({result.reason})"
        ]
        return {"guardrail": result, "pii_flags": pii_flags, "bad_citations": bad_citations, "trace": trace}

    def route_after_router(state: AgentState) -> str:
        return state["route"]

    def route_after_retrieval(state: AgentState) -> str:
        """Gate synthesis on retrieval actually covering the question."""
        if state.get("top_score", 0.0) < settings.min_retrieval_score:
            return "refuse"
        if state.get("coverage", 0.0) < settings.min_query_coverage:
            return "refuse"
        return "synthesize"

    def route_after_critic(state: AgentState) -> str:
        if state["guardrail"].passed:
            return "accept"
        if state.get("attempts", 0) >= settings.max_synthesis_attempts:
            return "give_up"
        return "retry"

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("direct", direct_node)
    graph.add_node("refuse", refuse_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_after_router, {"retrieve": "retriever", "direct": "direct"})
    graph.add_conditional_edges(
        "retriever",
        route_after_retrieval,
        {"synthesize": "synthesizer", "refuse": "refuse"},
    )
    graph.add_edge("synthesizer", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"accept": END, "retry": "synthesizer", "give_up": END},
    )
    graph.add_edge("direct", END)
    graph.add_edge("refuse", END)

    return graph.compile()


def run_query(vector_store: VectorStore, query: str, llm: LLM | None = None) -> AgentState:
    """Run a single query through the agent graph.

    The resulting state carries a ``metrics`` key holding the per-node
    telemetry for this request.
    """
    compiled = build_graph(vector_store, llm=llm)
    initial_state: AgentState = {"query": query, "attempts": 0, "trace": []}

    with collect_metrics() as metrics:
        result = compiled.invoke(initial_state)

    result["metrics"] = metrics
    return result
