"""The multi-agent workflow, built as a LangGraph state graph.

Agents (graph nodes):

    router       -> decides whether the query needs retrieval at all
                    (e.g. "hi" doesn't; "what is RAG?" does)
    retriever    -> pulls the top-k relevant chunks from the vector store
    synthesizer  -> drafts an answer grounded in the retrieved context
    critic       -> a guardrail agent that scores the draft for groundedness;
                    on failure it routes back to the synthesizer with
                    feedback (bounded by max_synthesis_attempts) instead of
                    silently returning an ungrounded answer
    direct       -> answers small-talk / meta queries without retrieval

This "generate -> critique -> retry" loop is the same self-correction
pattern used in production agentic RAG systems to keep answers honest
without a human in the loop for every request.
"""
from __future__ import annotations

from typing import List, TypedDict

from langgraph.graph import END, StateGraph

from .config import settings
from .guardrails import GuardrailResult, check_grounding, check_pii_leak
from .ingest import Chunk
from .llm import LLM, get_llm
from .vectorstore import RetrievedChunk, VectorStore

SMALL_TALK_MARKERS = ("hi", "hello", "hey", "thanks", "thank you", "who are you")


class AgentState(TypedDict, total=False):
    query: str
    route: str
    retrieved: List[RetrievedChunk]
    answer: str
    guardrail: GuardrailResult
    pii_flags: List[str]
    attempts: int
    trace: List[str]


def build_graph(vector_store: VectorStore, llm: LLM | None = None):
    """Compile the LangGraph agent graph, bound to a given vector store / LLM."""
    llm = llm or get_llm(settings.llm_backend)

    def router_node(state: AgentState) -> AgentState:
        query = state["query"].strip().lower()
        is_small_talk = len(query.split()) <= 3 and any(m in query for m in SMALL_TALK_MARKERS)
        route = "direct" if is_small_talk else "retrieve"
        trace = state.get("trace", []) + [f"router -> {route}"]
        return {"route": route, "trace": trace}

    def retriever_node(state: AgentState) -> AgentState:
        results = vector_store.search(state["query"], k=settings.top_k)
        trace = state.get("trace", []) + [f"retriever -> {len(results)} chunk(s)"]
        return {"retrieved": results, "trace": trace}

    def synthesizer_node(state: AgentState) -> AgentState:
        retrieved: List[RetrievedChunk] = state.get("retrieved", [])
        chunks: List[Chunk] = [r.chunk for r in retrieved]
        feedback = None
        if state.get("guardrail") and not state["guardrail"].passed:
            feedback = state["guardrail"].reason
        answer = llm.generate(state["query"], chunks, feedback=feedback)
        attempts = state.get("attempts", 0) + 1
        trace = state.get("trace", []) + [f"synthesizer -> draft #{attempts} via {llm.backend_name}"]
        return {"answer": answer, "attempts": attempts, "trace": trace}

    def direct_node(state: AgentState) -> AgentState:
        trace = state.get("trace", []) + ["direct -> small-talk response"]
        return {"answer": "Hello! Ask me a question and I'll answer it using the ingested knowledge base.", "trace": trace}

    def critic_node(state: AgentState) -> AgentState:
        retrieved: List[RetrievedChunk] = state.get("retrieved", [])
        chunks = [r.chunk for r in retrieved]
        result = check_grounding(state["answer"], chunks, threshold=settings.grounding_threshold)
        pii_flags = check_pii_leak(state["answer"])
        trace = state.get("trace", []) + [
            f"critic -> passed={result.passed} score={result.grounding_score} ({result.reason})"
        ]
        return {"guardrail": result, "pii_flags": pii_flags, "trace": trace}

    def route_after_router(state: AgentState) -> str:
        return state["route"]

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

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_after_router, {"retrieve": "retriever", "direct": "direct"})
    graph.add_edge("retriever", "synthesizer")
    graph.add_edge("synthesizer", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"accept": END, "retry": "synthesizer", "give_up": END},
    )
    graph.add_edge("direct", END)

    return graph.compile()


def run_query(vector_store: VectorStore, query: str, llm: LLM | None = None) -> AgentState:
    """Convenience wrapper: build (or reuse) the graph and run a single query."""
    compiled = build_graph(vector_store, llm=llm)
    initial_state: AgentState = {"query": query, "attempts": 0, "trace": []}
    return compiled.invoke(initial_state)
