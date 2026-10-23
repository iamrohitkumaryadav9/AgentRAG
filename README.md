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
  (ruff lint, then pytest across Python 3.10, 3.12 and 3.14), then the eval harness, which fails the build if
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
| Latency p50 / p95 | ~14 ms / ~15 ms |
| Mean est. tokens per query | 610 |

## Retrieval ablation (BEIR SciFact)

The 12-question golden set above checks the pipeline; it cannot rank retrieval strategies. For that, the
repo includes an ablation on the public **BEIR SciFact** benchmark (5,183 abstracts, 300 test claims):

| System | nDCG@10 | Recall@5 | Recall@100 | Latency p50 (ms) |
| --- | --- | --- | --- | --- |
| TF-IDF (this app's current retriever) | 0.596 | 0.667 | 0.892 | 49 |
| BM25 | 0.670 | 0.728 | 0.886 | 18 |
| Dense: all-MiniLM-L6-v2 | 0.645 | 0.738 | 0.925 | 18 |
| Dense: bge-small-en-v1.5 | **0.713** | 0.759 | 0.942 | 32 |
| Hybrid RRF: BM25 + MiniLM | 0.688 | 0.757 | 0.961 | 35 |
| Hybrid RRF: BM25 + bge-small | 0.711 | **0.762** | **0.965** | 58 |
| Hybrid (bge) + cross-encoder rerank | 0.695 | 0.748 | 0.965 | 6,159 |

- BM25 over TF-IDF is a clear gain (+0.074 nDCG@10, 95% CI [+0.046, +0.102]).
- Hybrid (bge) beats BM25 by +0.041 [+0.018, +0.064], significant after multiple-comparison correction, but
  is **statistically indistinguishable from bge-small alone** on nDCG@10 (−0.002 [−0.029, +0.026]). What hybrid
  adds is deeper recall (Recall@100 0.965 vs 0.942).
- The off-the-shelf MS MARCO cross-encoder did **not** help (−0.016, not significant) and costs ~6 s/query on CPU.
- Metric code is validated against published BM25 figures and the MTEB scores of both dense models.

Full tables, confidence intervals, caveats and reproduction steps: [docs/scifact_ablation.md](docs/scifact_ablation.md).
Where hybrid loses to BM25 and why: [docs/scifact_error_analysis.md](docs/scifact_error_analysis.md).
The app's default retrieval backend has **not** been changed to reflect these results yet.

## Quickstart

Requires **Python 3.10+** (developed and verified locally on 3.14; CI runs
3.10, 3.12 and 3.14). Use a Python version that has prebuilt wheels for the
dependencies -- a brand-new interpreter release may not, and `pip` will then
try to compile packages from source.

```bash
git clone https://github.com/iamrohitkumaryadav9/AgentRAG.git
cd AgentRAG
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Runs fully offline: TF-IDF retrieval + extractive generation, no API key
python cli.py "What is retrieval-augmented generation?"
```

Sample output:

Real output with the offline defaults (answer text wrapped for readability;
timings vary by machine):

```
Indexing 'data/sample_docs' ...
Indexed 10 chunks  |  retrieval=tfidf  generation=extractive

Q: What is retrieval-augmented generation?

A: # Retrieval-Augmented Generation Retrieval-Augmented Generation, or RAG, is
   a technique that combines a retrieval system with a language model. The
   main risks are retrieval failure (the right passage is never found), poor
   chunking (context is cut in the wrong place), and generation that drifts
   away from the retrieved context ...

Confidence: 0.7
Citations: rag_systems.md#0, rag_systems.md#2
Guardrail: PASSED  (grounding_score=1.0, sufficient overlap with retrieved context)
Attempts: 1

Sources:
  - rag_systems.md (score=0.377)
  - rag_systems.md (score=0.060)
  - responsible_ai.md (score=0.024)

Trace:
  * router -> retrieve
  * retriever -> 3 chunk(s), top_score=0.3773, coverage=1.0
  * synthesizer -> draft #1 via extractive
  * critic -> passed=True score=1.0 (sufficient overlap with retrieved context)

Telemetry (total 38.04 ms, ~693 tokens):
  * router: 0.12 ms
  * retriever: 37.22 ms
  * synthesizer: 0.28 ms
  * critic: 0.42 ms
```

The extractive generator stitches together the best-matching sentences, so
answers can include raw markdown headings and read choppily -- that is the
offline fallback, not the ceiling. Enable Gemini (below) for fluent answers.

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

### Reproducing the retrieval ablation (optional, heavy)

```bash
pip install -e ".[bench]"      # pinned torch / sentence-transformers / faiss-cpu / rank-bm25
python scripts/run_ablation.py --systems tfidf,bm25,minilm,bge-small,hybrid-minilm,hybrid-bge,hybrid-bge-rerank
python scripts/run_significance.py && python scripts/run_error_analysis.py
```

CPU-only this takes roughly an hour, mostly encoding the corpus and running the reranker. CI does not run it.

### Running the tests

```bash
pytest -q
ruff check .
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
  bench/            SciFact retrieval ablation: metrics, bootstrap stats, systems, error analysis
cli.py              terminal demo
scripts/            index builder, eval runner, ablation / significance / error-analysis runners
tests/              pytest suite (offline, no external calls)
docs/               SciFact ablation write-up, error analysis, committed result files
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
