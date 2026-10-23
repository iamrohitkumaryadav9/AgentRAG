# Retrieval ablation on BEIR SciFact

Seven retrieval strategies compared on the BEIR SciFact test set (5,183 abstracts, 300 claim queries),
with confidence intervals on every headline difference. Everything here is reproducible from this repo.

## Results

| System | nDCG@10 | MRR@10 | Recall@5 | Recall@50 | Recall@100 |
| --- | --- | --- | --- | --- | --- |
| TF-IDF (the app's previous default) | 0.596 | 0.553 | 0.667 | 0.870 | 0.892 |
| BM25 | 0.670 | 0.639 | 0.728 | 0.876 | 0.886 |
| Dense: all-MiniLM-L6-v2 | 0.645 | 0.605 | 0.738 | 0.892 | 0.925 |
| Dense: bge-small-en-v1.5 | 0.713 | 0.682 | 0.759 | 0.925 | 0.942 |
| Hybrid RRF: BM25 + MiniLM | 0.688 | 0.653 | 0.757 | 0.935 | 0.961 |
| Hybrid RRF: BM25 + bge-small | 0.711 | 0.682 | 0.762 | 0.947 | 0.965 |
| Hybrid (bge) + cross-encoder rerank | 0.695 | 0.664 | 0.748 | 0.947 | 0.965 |

Raw numbers: [results/scifact_summary.json](results/scifact_summary.json) (its latency fields come from the
first ablation run; the latency section below supersedes them).

## Latency

Measured on an Intel Core i5-10210U (4 cores / 8 threads), CPU only, Python 3.14.6, PyTorch 2.14.1+cpu, with
`scripts/run_latency.py`. Three execution modes, with the same retrieval results in all of them (nDCG@10
identical to four decimals, so batching changes speed, not answers):

| System | Queries | Sequential, 1 thread: p50 / p95 | Sequential, default threads: p50 / p95 | Batched, default threads: ms per query (queries/s) |
| --- | --- | --- | --- | --- |
| TF-IDF | 300 | 58 / 70 ms | 63 / 91 ms | 63 (15.8) |
| BM25 | 300 | 15 / 29 ms | 15 / 29 ms | 16 (60.6) |
| Dense: MiniLM | 300 | 20 / 27 ms | 19 / 24 ms | 10 (102.0) |
| Dense: bge-small | 300 | 45 / 60 ms | 40 / 51 ms | 22 (45.5) |
| Hybrid: BM25 + MiniLM | 300 | 39 / 62 ms | 42 / 66 ms | 27 (37.7) |
| Hybrid: BM25 + bge-small | 300 | 62 / 92 ms | 57 / 89 ms | 38 (26.2) |
| Hybrid (bge) + cross-encoder | 30 | 10,584 / 12,670 ms | 6,263 / 8,103 ms | 5,796 (0.2) |

- **What each mode means.** "Sequential" is one query at a time, which is what serving a single request looks
  like. "Default threads" is PyTorch's 4 intra-op threads and FAISS's 8 OpenMP threads. "1 thread" limits both
  libraries to one thread. "Batched" encodes all queries in batches and runs one FAISS search (and, for the
  reranker, one cross-encoder pass over every query–candidate pair); it is the offline throughput number,
  reported as amortised time per query.
- **My first latency numbers were mislabelled.** The earlier version of this document called the sequential
  figures "single-threaded". They were not: PyTorch and FAISS were using their default 4 and 8 threads. The
  true single-thread column above is new.
- **Thread count barely matters for the small encoders** (within about ±15%, which is also the noise level).
  It matters only for the reranker, which is 1.7× slower on one thread.
- **Batching roughly doubles dense throughput** (MiniLM 1.9×, bge-small 1.8×, hybrid 1.5–1.6×), because
  encoding many queries at once amortises per-call overhead. BM25 and TF-IDF do not benefit: their scoring
  loops are single-threaded Python/NumPy and were not parallelised.
- **The cross-encoder is compute-bound.** Batching gains only 1.08×; it costs about 5.8 s per query even in
  bulk (50 pairs of up to 512 tokens each): about 110× the hybrid retriever's per-query time sequentially
  and about 150× batched. This is the cost of a gain that, on this benchmark, did not materialise.
- **The reranker row uses 30 queries**, not 300, because a single-thread run over 300 would take hours; its
  percentiles are therefore rougher. Its nDCG@10 on those 30 (0.709) is not comparable with the 300-query figure.
- **Run-to-run noise is large.** The sequential default-thread measurements differ from my first ablation run
  by up to 29% for the same configuration (TF-IDF: 49 ms then 63 ms; bge-small: 32 then 40 ms). Use ratios and
  orders of magnitude, not exact milliseconds.

## Are the differences real? (paired bootstrap on per-query nDCG@10)

10,000 query resamples, seeded. The 95% interval is for the mean per-query nDCG@10 difference (A − B).
"p" is the two-sided bootstrap p-value (resolution 0.0001); "p (Holm)" corrects for testing all 21 system
pairs at once, so headline comparisons are not cherry-picked from a larger family.

| A vs B | mean diff | 95% CI | p | p (Holm) |
| --- | --- | --- | --- | --- |
| BM25 vs TF-IDF | +0.074 | [+0.046, +0.102] | 0.0001 | 0.0021 |
| MiniLM vs BM25 | −0.025 | [−0.063, +0.013] | 0.1928 | 0.9640 |
| bge-small vs BM25 | +0.043 | [+0.007, +0.078] | 0.0208 | 0.2288 |
| bge-small vs MiniLM | +0.068 | [+0.041, +0.094] | 0.0001 | 0.0021 |
| Hybrid (MiniLM) vs BM25 | +0.018 | [−0.002, +0.039] | 0.0828 | 0.6624 |
| Hybrid (MiniLM) vs MiniLM | +0.043 | [+0.015, +0.071] | 0.0036 | 0.0432 |
| Hybrid (bge) vs BM25 | +0.041 | [+0.018, +0.064] | 0.0006 | 0.0084 |
| Hybrid (bge) vs bge-small | −0.002 | [−0.029, +0.026] | 0.9194 | 1.0000 |
| Rerank vs Hybrid (bge) | −0.016 | [−0.045, +0.014] | 0.2978 | 0.9720 |
| Rerank vs bge-small | −0.017 | [−0.045, +0.011] | 0.2430 | 0.9720 |

All 21 pairs: [results/scifact_significance.json](results/scifact_significance.json). The BEIR-paper
replication (item 5 below) is deliberately outside this 21-pair family, so adding a sanity check never changes
the corrected p-values of the headline comparisons; it is reported uncorrected as a single pre-declared test.

## What the numbers support

1. **Replacing TF-IDF with BM25 is a clear win** (+0.074 nDCG@10, Holm p = 0.002) at lower latency.
2. **Hybrid (BM25 + bge-small) beats BM25 by +0.041** [0.018, 0.064], and that survives the multiple-
   comparison correction (Holm p = 0.008). **bge-small alone beats BM25 by a similar +0.043, but its interval
   barely clears zero** [0.007, 0.078] and it does **not** survive correction (Holm p = 0.229). The two
   top systems cannot be separated: hybrid vs bge-small is −0.002 [−0.029, +0.026].
3. **What hybrid buys is recall depth, not top-10 precision.** Hybrid (bge) reaches Recall@100 of 0.965 versus
   0.942 (bge-small) and 0.886 (BM25), and Recall@50 of 0.947 versus 0.925 and 0.876. I did not run a
   significance test on the recall differences, so treat them as observed, not established. For a pipeline
   that feeds a reranker or an LLM from a wide candidate pool, that ceiling matters; for ranking the first
   ten results it does not help here.
4. **The model matters more than the fusion.** bge-small beats MiniLM by +0.068 (Holm p = 0.002);
   fusing BM25 into MiniLM helps it (+0.043, Holm p = 0.043, a marginal result) but it stays numerically
   below bge-small (0.688 vs 0.713; I did not single that pair out for discussion).
5. **The cross-encoder rerank did not help, and the result is consistent with the BEIR paper.** On top of the
   hybrid it scored −0.016 nDCG@10 versus its own first stage (not significant: [−0.045, +0.014]). To check
   that this is not an implementation problem, I replicated the paper's own configuration (BM25 top-100 →
   the same 6-layer MS MARCO MiniLM cross-encoder):

   | Setup | First-stage nDCG@10 | After reranking | Change |
   | --- | --- | --- | --- |
   | BEIR paper: BM25 → cross-encoder, top-100 | 0.665 | 0.688 | +0.023 |
   | This repo, same setup | 0.670 | 0.680 | +0.010 [−0.020, +0.040] (not significant) |
   | This repo: hybrid (BM25 + bge-small) → cross-encoder, top-50 | 0.711 | 0.695 | −0.016 [−0.045, +0.014] (not significant) |

   The paper's figures are in its main results table ([Thakur et al., BEIR](https://arxiv.org/abs/2104.08663),
   page 7: SciFact BM25 0.665, BM25+CE 0.688; I read them from the paper, not a secondary source). My
   replication lands 0.008 below the paper's 0.688; its gain over my BM25 is smaller than the paper's, though
   the interval includes the paper's +0.023. I did not investigate the gap (candidates: the first-stage BM25
   implementation, input formatting, truncation).

   **The pattern: reranked quality sits at about 0.68–0.70 whether the first stage scores 0.670 or 0.711.** A
   weak first stage gains a little; a strong one gives some back. With this reranker, the ceiling on SciFact
   appears to be set by the reranker, not by candidate quality, so a better first stage does not make it pay
   off. That is a description of three data points, not a demonstrated mechanism; I have not tested why an
   MS MARCO cross-encoder plateaus here. (An earlier draft suggested domain shift. That was untested, so it
   has been removed.) Its Recall@50 and @100 equal the first stage's by construction, since it only reorders
   the top of the list. Cost is covered in the Latency section above.

## Reproducing and validating the setup

- **BM25 sanity check (done before building anything else).** My BM25 reached nDCG@10 = 0.670. Published
  figures are 0.665 (BEIR paper), 0.673 (a Vespa write-up) and 0.683 (Pyserini 0.22.1, in an arXiv
  paper). Mine falls inside that band.
- **Dense models verified against MTEB.** Source data for the MTEB leaderboard:
  - bge-small-en-v1.5: the model card's MTEB metadata lists SciFact nDCG@10 = 71.275, Recall@5 = 75.983,
    Recall@100 = 94.167, MRR@10 = 68.219. Mine: 0.7127, 0.7590, 0.9417, 0.6822.
  - all-MiniLM-L6-v2: the MTEB results repository lists nDCG@10 = 0.64508, Recall@5 = 0.73794,
    Recall@100 = 0.925, MRR@10 = 0.6047. Mine: 0.6451, 0.7379, 0.925, 0.6047.
  - All eight values agree to within 0.001; the largest gap is bge-small Recall@5 (0.7590 vs 0.7598). That
    supports the metric code and the encoding setup being correct.
- **The shipped code scores like the benchmarked code.** The app's own backends (not the benchmark's
  re-implementations) were run through the same harness:

  | App backend | nDCG@10 | Recall@100 | Benchmark counterpart | p50 latency |
  | --- | --- | --- | --- | --- |
  | `bm25` (scikit-learn/scipy, Lucene-style idf) | 0.6694 | 0.8859 | `rank_bm25`: 0.6701 / 0.8859 | 0.5 ms (vs 15 ms) |
  | `hybrid` (BM25 + bge-small, RRF) | 0.7103 | 0.9650 | benchmark hybrid: 0.7110 / 0.9650 | 58 ms |

  The ≤ 0.001 nDCG differences come from the idf variant. The app's BM25 is also much faster because it
  precomputes per-term weights; that speed-up does not carry over to the hybrid, whose time is dominated by
  encoding the query (the app's hybrid was a single sequential run, with the usual run-to-run noise).
- Metric definitions have hand-computed unit tests (`tests/test_bench_metrics.py`).

```bash
pip install -e ".[bench]"                       # pinned: torch, sentence-transformers, faiss-cpu, rank-bm25
python scripts/run_ablation.py --systems tfidf,bm25,minilm,bge-small,hybrid-minilm,hybrid-bge,hybrid-bge-rerank
python scripts/run_ablation.py --systems bm25-rerank100      # replication of the BEIR paper's BM25+CE
python scripts/run_significance.py
python scripts/run_error_analysis.py
python scripts/run_latency.py                                # three execution modes
python scripts/run_complementarity.py
```

The first run downloads SciFact and the model weights; encoding the corpus on CPU took about 7 minutes
(MiniLM) and 20 minutes (bge-small), and is cached afterwards. The reranker run takes roughly 30–40 minutes.

## Method details

- **Data:** BEIR SciFact, test split, 300 queries, graded labels treated as binary relevance (> 0).
  Documents are `title + " " + abstract`.
- **Metrics:** BEIR protocol. Recall@k = relevant retrieved in top k ÷ all relevant; MRR and nDCG at 10.
- **TF-IDF:** the app's own `TfidfVectorStore` (1–2-grams, English stop words).
- **BM25:** `rank_bm25` Okapi, k1 = 1.5, b = 0.75, lower-cased `\w+` tokens, scikit-learn English stop words.
- **Dense:** sentence-transformers, L2-normalised embeddings, exact FAISS inner-product search. bge-small
  uses its recommended query instruction ("Represent this sentence for searching relevant passages: ");
  MiniLM uses none.
- **Hybrid:** reciprocal rank fusion, score = Σ 1 / (60 + rank), over each retriever's top 100.
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` rescoring the hybrid's top 50; ranks 51–100 keep
  their first-stage order.

## Limitations

- **One dataset, 300 queries.** SciFact is small, scientific, and claim-based. Rankings of these systems
  may differ on other domains; it says nothing about the app's own three-document sample corpus.
- **Latency is rough.** One CPU-only Windows laptop, after a 5-query warm-up; repeat runs differ by up to
  about 30%. Hybrid latency is the *sum* of its parts because the retrievers run sequentially in this code.
  No GPU numbers. See the Latency section for modes and caveats.
- **Defaults, not tuning.** BM25 parameters, the RRF constant (60) and the rerank depth (50) were not
  tuned. Weighted fusion or a different constant could change the hybrid results.
- **The app default follows this study** (`auto` → hybrid BM25 + bge-small, falling back to BM25). That is a
  judgement from recall depth and complementary failure modes, not from a significant nDCG@10 gain over
  bge-small alone; see the rationale in the README. The app's BM25 uses Lucene-style idf, whereas the
  benchmark's BM25 used `rank_bm25`; the app's own classes were run through the same harness (see below).
- **Failure analysis:** see [scifact_error_analysis.md](scifact_error_analysis.md) for the 20 queries where
  hybrid loses to BM25.

## Future work

- **Weighted or tuned fusion, without leaking the test set.** Weighted RRF, the RRF constant, and the rerank
  depth should be tuned on SciFact's **train** split (809 claims in the original release, with relevance
  judgments in the BEIR download's `qrels/train.tsv`), chosen once, and then evaluated a single time on the
  300 test queries. Tuning on the test queries would leak and inflate the result, which is why it was not done
  here.
- A reranker trained or fine-tuned for scientific claims, and a significance test on the Recall@k
  differences.
