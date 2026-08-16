# AgentRAG — A Self-Correcting Multi-Agent RAG Assistant

AgentRAG is a small retrieval-augmented generation (RAG) service built around
a **multi-agent workflow**, not a single prompt. A router agent decides
whether a query needs retrieval, a retriever agent searches a vector store
and gates on whether the corpus actually covers the question, a synthesizer
agent drafts a **structured, schema-validated** answer, and a critic agent
checks it for groundedness and fabricated citations — looping back with
feedback if the check fails.

It ships with the two things that separate a RAG demo from a RAG system:
an **evaluation harness** that scores answer quality against a golden set,
and **OpenTelemetry instrumentation** that makes per-agent latency and cost
visible. Both run in CI as quality gates on every push.

Everything runs **fully offline out of the box** (TF-IDF retrieval + a
deterministic extractive generator), so you can clone it and try it in
seconds with no API keys — while every component is pluggable into a real
stack: dense embeddings (sentence-transformers + FAISS), a real LLM
(Google Gemini), and a FastAPI service.

```
            ┌────────┐
   query ──▶│ router │──── small talk ──────────────────────┐
            └───┬────┘                                      │
                │ needs retrieval                            ▼
                ▼                                        ┌────────┐
          ┌────────────┐                                 │ direct │──▶ answer
          │ retriever  │  FAISS / TF-IDF                 └────────┘
          └─────┬──────┘
                │
        coverage│< floor ──────────────▶ ┌────────┐
                │                        │ refuse │──▶ "I don't know"
                │ covers the question    └────────┘
                ▼
          ┌─────────────┐  fails groundedness / citation check (bounded retry)
          │ synthesizer │◀────────────────────────────────┐
          │ (structured │                                 │
          │   output)   │                                 │
          └─────┬───────┘                                 │
                ▼                                          │
          ┌───────────┐    passes ──────────────▶ answer + citations
          │  critic   │────────────────────────────────────┘
          │(guardrail)│
          └───────────┘
```

## Why this exists

I built this to go deeper into agentic AI systems, RAG pipelines, and
production AI-quality practice than a single notebook demo shows. It
deliberately mirrors the trade-offs a real AI-native team deals with: what
happens when the ideal embedding model isn't reachable, what happens when
the LLM key isn't configured, how you stop a multi-agent loop from either
hallucinating or looping forever, and how you know — with numbers — whether
a change made answers better or worse.

## Key design decisions

- **Multi-agent graph, not a single call.** Built with
  [LangGraph](https://github.com/langchain-ai/langgraph) as an explicit
  state graph (`agentrag/graph.py`): `router → retriever → synthesizer →
  critic`, with conditional edges for refusal and for bounded
  critic-driven retries.
- **Structured outputs, not prose parsing.** The synthesizer returns a
  Pydantic-validated `AnswerPayload` (answer + citations + confidence,
  `agentrag/schemas.py`). Because citations are structured, the critic can
  check them *mechanically* and catch a fabricated source — a failure mode
  that's dangerous precisely because the answer still looks well-attributed.
- **Guardrails as a first-class agent** (`agentrag/guardrails.py`):
  groundedness scoring, citation validation, a PII-leak scan, and an
  "honest refusal always passes" rule.
- **Refuses instead of guessing.** A relevance gate compares the question's
  topical terms against the retrieved context and abstains when coverage is
  too low. See the note below on why this replaced a similarity threshold.
- **Evaluated, not vibes-checked** (`agentrag/evaluation.py`): a golden set
  of in-scope *and* deliberately out-of-scope questions, scored on retrieval
  hit-rate, groundedness, keyword recall, refusal accuracy, citation
  validity, p50/p95 latency, and estimated tokens per query.
- **Instrumented with OpenTelemetry** (`agentrag/observability.py`): every
  agent node emits a span with latency, retrieval quality, retry depth,
  grounding score, and a token-cost estimate. Spans are summarised per
  request by default, and can be shipped to any OTLP collector by setting
  two env vars — no code change.
- **Every external dependency degrades gracefully.** Dense embeddings when
  available, TF-IDF otherwise; Gemini when `GOOGLE_API_KEY` is set, an
  offline extractive generator otherwise. The pipeline never hard-fails
  because an optional dependency is missing.
- **CI enforces both gates** (`.github/workflows/ci.yml`): the test suite
  across Python 3.10–3.12, then the eval harness, which fails the build if
  answer quality drops below threshold. Runs with no secrets and no network.

### A finding from the eval harness

The first version of the eval reported **50% refusal accuracy**: asked
"Who won the 2026 FIFA World Cup final?", the system answered from weak
matches instead of declining, because the word *final* appears in the
corpus. This is exactly the failure an accuracy-only metric hides.

The first fix — a minimum similarity score — passed at one chunk size and
broke at another, because similarity scores shift with chunk size, corpus
size, and retrieval backend. The fix that held is **query-term coverage**:
the share of the question's topical terms actually present in the retrieved
context. It's normalised to [0,1] by construction, so it behaves
consistently across backends. On this corpus in-scope questions score ≥0.50
and out-of-scope ≤0.25, and `test_refusal_behaviour_is_stable_across_chunk_configs`
pins that property at three different chunk sizes.

Refusal accuracy is now 100% and enforced as a CI threshold.

## Current metrics

Measured by `scripts/run_eval.py` on the built-in corpus with the offline
backends (reproduce with `python scripts/run_eval.py`):

| Metric | Value |
| --- | --- |
| Retrieval hit-rate@k | 100.0% |
| Groundedness pass rate | 100.0% |
| Keyword recall | 80.0% |
| Refusal accuracy (out-of-scope) | 100.0% |
| Citation validity | 100.0% |
| Latency p50 / p95 | ~10 ms / ~12 ms |
| Mean est. tokens per query | 610 |

## Quickstart

```bash
git clone https://github.com/iamrohitkumaryadav9/AgentRAG.git
cd AgentRAG
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Runs fully offline: TF-IDF retrieval + extractive generation, no API key
python cli.py "What is retrieval-augmented generation?"
```

Sample output:

```
Q: What is retrieval-augmented generation?
A: Retrieval-Augmented Generation, or RAG, is a technique that combines a
   retrieval system with a language model...

Confidence: 0.85
Citations: rag_systems.md#0
Guardrail: PASSED  (grounding_score=1.0, sufficient overlap with retrieved context)
Attempts: 1

Telemetry (total 10.4 ms, ~610 tokens):
  * router: 0.02 ms
  * retriever: 3.1 ms
  * synthesizer: 0.9 ms
  * critic: 0.4 ms
```

### Running the evaluation harness

```bash
python scripts/run_eval.py                      # prints metrics, exits non-zero if below threshold
python scripts/run_eval.py --json-out eval.json --markdown-out eval.md
```

### Enabling real LLM generation (optional)

```bash
cp .env.example .env          # set GOOGLE_API_KEY
pip install -e ".[gemini]"
python cli.py "How do guardrails prevent hallucination?"
```

### Enabling dense semantic retrieval (optional)

```bash
pip install -e ".[dense]"     # sentence-transformers + FAISS
```

### Running the API

```bash
uvicorn agentrag.api:app --reload
# POST /ingest {"directory": "data/sample_docs"}
# POST /query  {"question": "What is a multi-agent system?"}
# POST /eval   -> current AI quality metrics
```

### Exporting traces to a collector

```bash
export AGENTRAG_OTEL_OTLP=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
python cli.py "What is RAG?"
```

### Running the tests

```bash
pytest -q
```

## Project layout

```
agentrag/
  config.py         runtime settings (env-var driven)
  ingest.py         document loading + sentence-aware chunking
  vectorstore.py    pluggable retrieval: DenseVectorStore (FAISS) / TfidfVectorStore
  llm.py            pluggable generation: GeminiLLM / ExtractiveLLM
  schemas.py        Pydantic structured-output contracts
  guardrails.py     groundedness, citation, PII, and relevance checks
  observability.py  OpenTelemetry spans + per-request metrics
  evaluation.py     golden-set eval harness and quality thresholds
  graph.py          the LangGraph multi-agent workflow
  api.py            FastAPI service
cli.py              terminal demo
scripts/            index builder, eval runner
tests/              pytest suite (offline, no external calls)
data/sample_docs/   built-in knowledge base used by the demo, tests, and eval
.github/workflows/  CI: test matrix + AI quality gate
```

## Ideas for extending this

- Replace the lexical groundedness heuristic with an NLI or
  embedding-similarity faithfulness model.
- Add a tool-use agent node (calculator, web search) alongside retrieval.
- Swap the in-memory index for a persistent vector database (Chroma,
  Qdrant, pgvector) so `/ingest` doesn't rebuild on every call.
- Track eval metrics over time to detect quality drift, rather than only
  gating per-commit.

## License

MIT — see [LICENSE](LICENSE).
