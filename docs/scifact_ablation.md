# Retrieval ablation on BEIR SciFact

Seven retrieval strategies compared on the BEIR SciFact test set (5,183 abstracts, 300 claim queries),
with confidence intervals on every headline difference. Everything here is reproducible from this repo.

## Results

| System | nDCG@10 | MRR@10 | Recall@5 | Recall@50 | Recall@100 | Latency p50 / p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF (the app's current retriever) | 0.596 | 0.553 | 0.667 | 0.870 | 0.892 | 49 / 62 |
| BM25 | 0.670 | 0.639 | 0.728 | 0.876 | 0.886 | 18 / 43 |
| Dense: all-MiniLM-L6-v2 | 0.645 | 0.605 | 0.738 | 0.892 | 0.925 | 18 / 25 |
| Dense: bge-small-en-v1.5 | 0.713 | 0.682 | 0.759 | 0.925 | 0.942 | 32 / 43 |
| Hybrid RRF: BM25 + MiniLM | 0.688 | 0.653 | 0.757 | 0.935 | 0.961 | 35 / 55 |
| Hybrid RRF: BM25 + bge-small | 0.711 | 0.682 | 0.762 | 0.947 | 0.965 | 58 / 100 |
| Hybrid (bge) + cross-encoder rerank | 0.695 | 0.664 | 0.748 | 0.947 | 0.965 | 6,159 / 8,063 |

Raw numbers: [results/scifact_summary.json](results/scifact_summary.json).

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

All 21 pairs: [results/scifact_significance.json](results/scifact_significance.json).

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
5. **The cross-encoder rerank did not help and is very expensive.** It scored −0.016 nDCG@10 versus its own
   first stage (not significant: [−0.045, +0.014]) at a median of 6.2 s per query on CPU, about 105× the
   hybrid's 58 ms. Its Recall@50 and @100 equal the first stage's by construction (it only reorders the top
   50). I did not test why. A plausible but unverified reason is domain shift: the model
   (`cross-encoder/ms-marco-MiniLM-L-6-v2`) was trained on web-search queries, and SciFact queries are
   scientific claims. A domain-appropriate reranker, or a larger one, might behave differently.

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
- Metric definitions have hand-computed unit tests (`tests/test_bench_metrics.py`).

```bash
pip install -e ".[bench]"                       # pinned: torch, sentence-transformers, faiss-cpu, rank-bm25
python scripts/run_ablation.py --systems tfidf,bm25,minilm,bge-small,hybrid-minilm,hybrid-bge,hybrid-bge-rerank
python scripts/run_significance.py
python scripts/run_error_analysis.py
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
- **Latency is rough.** Single-threaded queries on one CPU-only Windows laptop, after a 5-query warm-up.
  Repeat runs varied by roughly ±20% (BM25 p50 measured 14.6 ms and 17.9 ms in two runs), and hybrid
  latency is the *sum* of its parts because the retrievers run sequentially. Use the ratios, not the
  absolute milliseconds.
- **Defaults, not tuning.** BM25 parameters, the RRF constant (60) and the rerank depth (50) were not
  tuned. Weighted fusion or a different constant could change the hybrid results.
- **The app is unchanged.** `AGENTRAG_VECTOR_BACKEND=auto` still prefers MiniLM and falls back to TF-IDF.
  This study suggests BM25 or bge-small would be better defaults; that change has not been made.
- **Failure analysis:** see [scifact_error_analysis.md](scifact_error_analysis.md) for the 20 queries where
  hybrid loses to BM25.
