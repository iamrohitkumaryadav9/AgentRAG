# AgentRAG — A Self-Correcting Multi-Agent RAG Assistant

AgentRAG is a small retrieval-augmented generation (RAG) service built around
a **multi-agent workflow**, not a single prompt. A router agent decides
whether a query needs retrieval, a retriever agent searches a vector store
and gates on whether the corpus actually covers the question, a synthesizer
agent drafts a **structured, schema-validated** answer, and a critic agent
checks it for groundedness and fabricated citations — looping back with
feedback if the check fails.

It ships with the things that separate a RAG demo from a RAG system:
a **retrieval benchmark** on the public BEIR SciFact set (seven strategies,
confidence intervals, error analysis), a small **regression harness** that
runs in CI against a 3-document toy corpus, and **OpenTelemetry
instrumentation** that makes per-agent latency and cost visible. The toy
harness is a smoke test of pipeline behaviour, not a quality claim; the
retrieval benchmark is where the quality evidence lives.

Everything runs **fully offline out of the box** (BM25 retrieval + a
deterministic extractive generator), so you can clone it and try it in
seconds with no API keys — while every component is pluggable into a real
stack: hybrid BM25 + dense retrieval (sentence-transformers + FAISS), a real
LLM (Google Gemini), and a FastAPI service.

```
            ┌────────┐
   query ──▶│ router │──── small talk ──────────────────────┐
            └───┬────┘                                      │
                │ needs retrieval                            ▼
                ▼                                        ┌────────┐
          ┌────────────┐                                 │ direct │──▶ answer
          │ retriever  │  BM25 + dense (RRF) / BM25      └────────┘
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
- **Regression-tested, and benchmarked separately** (`agentrag/evaluation.py`,
  `agentrag/bench/`): a 12-question golden set of in-scope *and* deliberately
  out-of-scope questions guards pipeline behaviour (refusals, citation
  validity, latency, token cost) on a toy corpus. Retrieval quality is measured
  on BEIR SciFact instead. Note that the harness's "groundedness" is a
  lexical word-overlap check, not a measure of faithfulness.
- **Instrumented with OpenTelemetry** (`agentrag/observability.py`): every
  agent node emits a span with latency, retrieval quality, retry depth,
  grounding score, and a token-cost estimate. Spans are summarised per
  request by default, and can be shipped to any OTLP collector by setting
  two env vars — no code change.
- **Every external dependency degrades gracefully.** Hybrid BM25 + dense
  retrieval when the `dense` extra is installed, plain BM25 otherwise (never
  a silent drop to a weaker retriever than BM25); Gemini when `GOOGLE_API_KEY` is set, an
  offline extractive generator otherwise. The pipeline never hard-fails
  because an optional dependency is missing.
- **CI runs two gates** (`.github/workflows/ci.yml`): the test suite
  (ruff lint, then pytest across Python 3.10, 3.12 and 3.14), then the toy-corpus smoke test, which fails the
  build if pipeline behaviour regresses (e.g. refusals stop working). Runs with no secrets and no network.

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

## Toy-corpus smoke test (not a quality claim)

`scripts/run_eval.py` runs 12 questions against the 3 built-in sample documents (10 chunks). CI uses it to catch
regressions. **Read the numbers below as "the pipeline still behaves", not as accuracy**: with 12 questions and a
generator that copies sentences from the retrieved context, near-perfect scores are close to guaranteed.
In particular, "groundedness" is the share of answer words that also appear in the retrieved text, which an
extractive generator satisfies by construction; it measures nothing about whether a claim is true or supported.
(Reproduce with `python scripts/run_eval.py`.)

| Metric | Value |
| --- | --- |
| Retrieval hit-rate@k | 100.0% |
| Groundedness pass rate (lexical overlap, see above) | 100.0% |
| Keyword recall | 80.0% |
| Refusal accuracy (out-of-scope) | 100.0% |
| Citation validity | 100.0% |
| Latency p50 / p95 | ~12 ms / 14–50 ms (three runs) |
| Mean est. tokens per query | 609 |

The quality metrics are identical on the `tfidf`, `bm25` and `hybrid` backends (the golden set is 12 questions over a
10-chunk corpus, so it checks the pipeline and refusal behaviour but cannot rank retrievers; see the SciFact
ablation below for that). Latency above is for `bm25`.

## Retrieval ablation (BEIR SciFact)

The 12-question golden set above checks the pipeline; it cannot rank retrieval strategies. For that, the
repo includes an ablation on the public **BEIR SciFact** benchmark (5,183 abstracts, 300 test claims):

| System | nDCG@10 | Recall@5 | Recall@100 | p50 latency, sequential (ms) | Batched (ms/query) |
| --- | --- | --- | --- | --- | --- |
| TF-IDF (the app's previous default) | 0.596 | 0.667 | 0.892 | 63 | 63 |
| BM25 | 0.670 | 0.728 | 0.886 | 15 | 16 |
| Dense: all-MiniLM-L6-v2 | 0.645 | 0.738 | 0.925 | 19 | 10 |
| Dense: bge-small-en-v1.5 | **0.713** | 0.759 | 0.942 | 40 | 22 |
| Hybrid RRF: BM25 + MiniLM | 0.688 | 0.757 | 0.961 | 42 | 27 |
| Hybrid RRF: BM25 + bge-small | 0.711 | **0.762** | **0.965** | 57 | 38 |
| Hybrid (bge) + cross-encoder rerank | 0.695 | 0.748 | 0.965 | 6,263 | 5,796 |

Latency: CPU only (4 cores), one query at a time with default threads, and batched for offline throughput. A true
single-thread run is in the docs; repeat runs differ by up to ~30%, so read ratios, not exact milliseconds.

- BM25 over TF-IDF is a clear gain (+0.074 nDCG@10, 95% CI [+0.046, +0.102]).
- Hybrid (bge) beats BM25 by +0.041 [+0.018, +0.064], significant after multiple-comparison correction, but
  is **statistically indistinguishable from bge-small alone** on nDCG@10 (−0.002 [−0.029, +0.026]). What hybrid
  adds is deeper recall (Recall@100 0.965 vs 0.942).
- The off-the-shelf MS MARCO cross-encoder did **not** help (−0.016, not significant) and costs ~6 s/query on
  CPU (batching barely helps: 5.8 s). Replicating the BEIR paper's own BM25 → cross-encoder setup gave 0.680
  against the paper's 0.688; across both setups reranked quality sits at about 0.68–0.70 whether the first stage
  scores 0.670 or 0.711, so a better first stage does not make this reranker pay off. (Scored only on the 188
  claims that have evidence, the rerank is +0.009, not significant; the deficit sits in the 112
  "not enough info" claims. See the docs' second view.)
- Metric code is validated against published BM25 figures and the MTEB scores of both dense models.

Full tables, confidence intervals, caveats and reproduction steps: [docs/scifact_ablation.md](docs/scifact_ablation.md).
Where hybrid loses to BM25 and why: [docs/scifact_error_analysis.md](docs/scifact_error_analysis.md).
### Why hybrid (BM25 + bge-small, RRF) is the default

`AGENTRAG_VECTOR_BACKEND=auto` selects **hybrid retrieval** when the `dense` extra is installed and falls back to
**BM25** (not TF-IDF) when it is not. CI pins `AGENTRAG_VECTOR_BACKEND=tfidf` and the extractive generator, so it
stays fast and download-free.

- **Recall ceiling.** Hybrid leaves the fewest queries with no gold document anywhere in the top 100: 10, versus
  17 for bge-small and 32 for BM25. It finds at least one gold document for 9 of the 17 queries dense misses
  entirely and 22 of the 32 BM25 misses. Everything downstream (reranker, synthesizer, critic) can only use what
  retrieval surfaces, so the deepest, most complementary candidate pool is the safest foundation.
- **A hedge against dense blind spots.** The error analysis shows the dense model never retrieving documents
  whose match rests on an exact rare term (gene symbols, acronyms); BM25 keeps those reachable. The honest limit:
  on queries containing a very rare term (document frequency ≤ 3, n = 142), hybrid versus bge-small alone is
  +0.029 nDCG@10 with 95% CI [−0.007, +0.065], not distinguishable at this sample size. The benefit is
  protection against failure cases, not a measured gain in rank quality.
- **What it does not buy.** Top-10 precision: hybrid ≈ bge-small (−0.002 [−0.029, +0.026]). bge-small alone is a
  defensible alternative if latency or the two-index footprint matters more than coverage.
- **Cost.** Roughly 3–4× BM25's per-query latency (57 ms versus 15 ms sequentially on this CPU; 38 ms per query
  when batched), plus a one-off ~130 MB model download. The BM25 fallback needs no extra dependencies or downloads.

Reproduce the numbers with `python scripts/run_complementarity.py`.

## Claim verification on SciFact (frozen configuration, run once)

Can a critic make an LLM judge better? The pipeline retrieves abstracts for a scientific claim, has Gemini
(`gemini-3.8-flash`) rule SUPPORT / CONTRADICT / NOT_ENOUGH_INFO with cited sentences, then lets a critic (citation
rules, a blind polarity check, an NLI-based recheck of "not enough info" answers) challenge the verdict and retry. It
was developed on 60 train claims, **frozen and pushed before the dev claims were touched**, and run once over all 300
SciFact dev claims (124 SUPPORT, 64 CONTRADICT, 112 NOT_ENOUGH_INFO), with the comparisons and Holm correction
declared in advance.

| Arm | Macro-F1 | κ vs human labels | CONTRADICT recall | Cost per claim |
| --- | --- | --- | --- | --- |
| NLI model alone (no LLM) | 0.499 | 0.288 | 0.84 (precision 0.35) | free |
| **B0: judge only** | **0.781** | **0.668** | 0.69 | $0.0019 |
| N: + NEI recheck only | 0.797 | 0.690 | 0.75 | $0.0027 |
| B1: + LLM polarity check + NEI recheck (main method) | 0.777 | 0.659 | 0.73 | $0.0030 |
| B2: + NLI polarity check + NEI recheck | 0.770 | 0.654 | 0.69 | $0.0030 |
| Oracle: judge given the gold abstracts | 0.791 | 0.683 | 0.69 | $0.0009 |

**The main method did not beat the judge alone.** Primary comparison B1 − B0: macro-F1 −0.004 [−0.029, +0.022] and
contradiction recall +0.047 [−0.016, +0.118], Holm-corrected p = 1.0. The only component with a positive estimate is
the NEI recheck (contradiction recall +0.062 [+0.014, +0.127]), which does not survive correction across the 16
declared tests (Holm p = 0.50). The polarity checks lowered accuracy slightly, because the judge rarely gets the
direction wrong (5 swapped verdicts in 300) and its dominant error, answering "not enough info" when evidence exists
(47 of 188 claims), is invisible to a polarity check. Giving the judge the gold abstracts changed little (+0.010), so
retrieval (the right document in the top 5 for 93% of claims) is not the bottleneck. Faithfulness of the cited
sentences, by the LLM checker: 0.954 (146 of 153 decisive verdicts); the checker itself recovers 64% of human-marked
entailments, so read that as relative.

**Limitations.** Gemini may have seen SciFact in pretraining, which could inflate absolute scores. With only 64
CONTRADICT claims, effects under about 0.05–0.06 cannot be resolved. One run, one model, one configuration. Thinking
cannot be disabled on this model, so it is logged rather than limited. The whole dev run cost about Rs 118 (1,121 API
requests) against hard caps enforced in code. Full tables, the train-subset development (including a prompt revision
that did not fix the bias), and every caveat: [docs/claim_verification.md](docs/claim_verification.md). Reproduce the
tables with no API calls: `python scripts/summarize_verification.py docs/results/verify_dev_frozen.json`; the response
cache is committed, so `--offline` reruns are free.

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

# Runs fully offline: BM25 retrieval + extractive generation, no API key
python cli.py "What is retrieval-augmented generation?"
```

Sample output:

Real output of a fresh `pip install -e ".[dev]"` install, where `auto` resolves to BM25 (answer text wrapped for
readability; timings vary by machine; with `.[dense]` installed the first line reads `retrieval=hybrid-rrf`):

```
Indexing 'data/sample_docs' ...
Indexed 10 chunks  |  retrieval=bm25  generation=extractive

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
  - rag_systems.md (score=6.495)
  - rag_systems.md (score=2.491)
  - responsible_ai.md (score=1.024)

Trace:
  * router -> retrieve
  * retriever -> 3 chunk(s), top_score=6.4953, coverage=1.0
  * synthesizer -> draft #1 via extractive
  * critic -> passed=True score=1.0 (sufficient overlap with retrieved context)

Telemetry (total 3.02 ms, ~693 tokens):
  * router: 0.28 ms
  * retriever: 1.38 ms
  * synthesizer: 0.54 ms
  * critic: 0.82 ms
```

(BM25 scores are unbounded relevance sums, not cosine similarities; the `hybrid` backend reports a normalised
score in (0, 1].)

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

### Enabling hybrid retrieval (recommended)

```bash
pip install -e ".[dense]"     # sentence-transformers + FAISS
python cli.py "How do guardrails prevent hallucination?"   # now retrieval=hybrid-rrf
```

With the extra installed, `auto` fuses BM25 with `BAAI/bge-small-en-v1.5` embeddings via reciprocal rank fusion
(model weights download from Hugging Face on first use; if that fails, `auto` logs a warning and uses BM25).
Pick a backend explicitly with `AGENTRAG_VECTOR_BACKEND=hybrid|bm25|dense|tfidf`; explicit choices never fall
back silently.

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
  vectorstore.py    pluggable retrieval: Hybrid (BM25 + dense, RRF) / Bm25 / Dense (FAISS) / Tfidf
  fusion.py         reciprocal rank fusion, shared by the app and the benchmarks
  llm.py            pluggable generation: GeminiLLM / ExtractiveLLM
  schemas.py        Pydantic structured-output contracts
  guardrails.py     groundedness, citation, PII, and relevance checks
  observability.py  OpenTelemetry spans + per-request metrics
  evaluation.py     golden-set eval harness and quality thresholds
  graph.py          the LangGraph multi-agent workflow
  api.py            FastAPI service
  bench/            SciFact retrieval ablation: metrics, bootstrap stats, systems, error analysis
  verify/           claim verification: judge, critic (rules, polarity check, NEI recheck), LangGraph retry loop,
                    cached rate-limited Gemini client, hard budget + tripwires, metrics
cli.py              terminal demo
scripts/            index builder, eval runner, retrieval ablation runners, run_verification.py (tune / freeze / run),
                    summarize_verification.py
tests/              pytest suite (offline, no external calls, fake Gemini transport)
docs/               SciFact ablation, error analysis, claim-verification write-up, committed results + response cache
data/sample_docs/   built-in knowledge base used by the demo, tests, and eval
.github/workflows/  CI: test matrix + AI quality gate
```

## Ideas for extending this

- Replace the lexical groundedness heuristic in the question-answering graph with the blind polarity check from the
  claim-verification pipeline (the LLM version recovered 64% of human-marked entailments on SciFact; NLI 54%).
- Add a tool-use agent node (calculator, web search) alongside retrieval.
- Swap the in-memory index for a persistent vector database (Chroma,
  Qdrant, pgvector) so `/ingest` doesn't rebuild on every call.
- Track eval metrics over time to detect quality drift, rather than only
  gating per-commit.

## License

MIT — see [LICENSE](LICENSE).
