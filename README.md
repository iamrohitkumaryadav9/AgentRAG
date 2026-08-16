# AgentRAG — A Self-Correcting Multi-Agent RAG Assistant

AgentRAG is a small retrieval-augmented generation (RAG) service built around
a **multi-agent workflow**, not a single prompt. A router agent decides
whether a query needs retrieval, a retriever agent searches a vector store,
a synthesizer agent drafts a grounded answer, and a critic agent
(a guardrail) checks the answer for groundedness before it's returned —
looping back to the synthesizer with feedback if the check fails.

It's built to run **fully offline out of the box** (TF-IDF retrieval +
a deterministic extractive "LLM") so anyone can clone it and try it in
seconds with no API keys, while every component is also pluggable into a
real production stack: dense embeddings (sentence-transformers + FAISS),
a real LLM (Google Gemini), and a FastAPI service.

```
            ┌────────┐
   query ──▶│ router │──── small talk ───────────────┐
            └───┬────┘                               │
                │ needs retrieval                     ▼
                ▼                                  ┌────────┐
          ┌────────────┐                           │ direct │──▶ answer
          │ retriever  │  (vector store: FAISS /    └────────┘
          │            │   TF-IDF fallback)
          └─────┬──────┘
                ▼
          ┌─────────────┐   fails groundedness check (retry, bounded)
          │ synthesizer │◀───────────────────────────────┐
          └─────┬───────┘                                │
                ▼                                         │
          ┌───────────┐   passes  ─────────────────▶ answer + sources
          │  critic   │───────────────────────────────────┘
          │(guardrail)│
          └───────────┘
```

## Why this exists

I built this to go deeper into agentic AI systems, RAG pipelines, and
production AI-quality patterns (grounding checks, guardrails,
observability via structured traces) beyond what a single Colab notebook
demo shows. It deliberately mirrors the kind of trade-offs a real AI-native
engineering team deals with: what happens when the "ideal" embedding model
isn't reachable, what happens when the LLM API key isn't configured, and
how do you stop a multi-agent loop from either hallucinating or looping
forever.

## Key design decisions

- **Multi-agent graph, not a single call.** Built with
  [LangGraph](https://github.com/langchain-ai/langgraph) as an explicit
  state graph (`agentrag/graph.py`): `router → retriever → synthesizer →
  critic`, with a bounded retry edge from `critic` back to `synthesizer`
  when the groundedness check fails.
- **Guardrails as a first-class agent**, not an afterthought
  (`agentrag/guardrails.py`): a lexical-overlap groundedness score, an
  "honest refusal is always acceptable" rule (never guess when there's no
  supporting context), and a basic PII-leak scan.
- **Every external dependency degrades gracefully.** Dense embeddings
  (sentence-transformers + FAISS) are used when available; if the model
  can't be downloaded (no network, offline CI, sandboxed environment) the
  system automatically falls back to a TF-IDF vector store so retrieval
  never hard-fails. Same pattern for generation: Gemini when
  `GOOGLE_API_KEY` is set, an offline extractive generator otherwise.
- **Sentence-aware chunking with overlap** (`agentrag/ingest.py`) instead of
  naive character-window splitting, so chunks stay semantically coherent.
- **Served via FastAPI** (`agentrag/api.py`) with `/ingest` and `/query`
  endpoints, so the pipeline is deployable, not just script-shaped.
- **Fully tested** (`tests/`) — chunking, both retrieval backends, guardrail
  pass/fail cases, and two full agent-graph runs — using the offline
  backends so CI needs no secrets or network access.

## Quickstart

```bash
git clone https://github.com/iamrohitkumaryadav9/AgentRAG.git
cd AgentRAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Runs fully offline: TF-IDF retrieval + extractive generation, no API key needed
python cli.py "What is retrieval-augmented generation?"
```

Sample output:

```
Q: What is retrieval-augmented generation?
A: Retrieval-Augmented Generation, or RAG, is a technique that combines a
   retrieval system with a language model... (Source: rag_systems.md)

Guardrail: PASSED  (grounding_score=1.0, sufficient overlap with retrieved context)
Attempts: 1
Sources:
  - rag_systems.md (score=0.349)
  ...
```

### Enabling real LLM generation (optional)

```bash
cp .env.example .env
# set GOOGLE_API_KEY=your-key in .env
python cli.py "How do guardrails prevent hallucination?"
```

### Running the API

```bash
uvicorn agentrag.api:app --reload
# POST /ingest {"directory": "data/sample_docs"}
# POST /query  {"question": "What is a multi-agent system?"}
```

### Running the tests

```bash
pytest -q
```

## Project layout

```
agentrag/
  config.py       runtime settings (env-var driven)
  ingest.py       document loading + sentence-aware chunking
  vectorstore.py  pluggable retrieval: DenseVectorStore (FAISS) / TfidfVectorStore
  llm.py          pluggable generation: GeminiLLM / ExtractiveLLM
  guardrails.py   groundedness + PII checks applied to every answer
  graph.py        the LangGraph multi-agent workflow
  api.py          FastAPI service
cli.py            terminal demo
scripts/          index-building / sanity-check helper
tests/            pytest suite (offline, no external calls)
data/sample_docs/ tiny built-in knowledge base used by the demo & tests
```

## Ideas for extending this

- Swap the lexical groundedness heuristic for an embedding-similarity or
  NLI-based faithfulness check.
- Add a `tool_use` agent node (e.g. a calculator or web-search tool) to the
  graph alongside retrieval.
- Add OpenTelemetry spans around each node in `graph.py` for real
  observability instead of the current plain-text trace list.
- Swap `TfidfVectorStore` for a persistent vector database (Chroma, Qdrant,
  pgvector) instead of rebuilding the index in memory on every `/ingest`
  call.

## License

MIT — see [LICENSE](LICENSE).
