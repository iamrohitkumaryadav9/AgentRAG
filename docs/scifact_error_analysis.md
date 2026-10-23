# Error analysis: where hybrid retrieval loses to BM25 on SciFact

**Setup.** Hybrid = reciprocal rank fusion (k=60) of BM25 and bge-small-en-v1.5, each contributing its
top 100. Comparison metric: per-query nDCG@10 on the 300 BEIR SciFact test queries.
Reproduce with `python scripts/run_error_analysis.py` (needs the cached runs from
`scripts/run_ablation.py`). The full 20-row table is in
[results/scifact_error_analysis.md](results/scifact_error_analysis.md).

## Context: hybrid is better overall

Hybrid beats BM25 on 64 queries, loses on 27, and ties on 209. The 27 losses sum to −7.96 nDCG@10 and the
64 wins to +20.23, a net +12.27 over 300 queries, i.e. the +0.041 mean difference reported in the main
results (95% CI [+0.018, +0.064], significant after Holm correction). This document explains the *minority*
of queries that go the other way. It is not evidence that hybrid is worse.

## How the 20 queries and groups were chosen

- The 20 queries are the **largest per-query losses** (hybrid nDCG@10 − BM25 nDCG@10), ties broken by query
  id. They are the worst cases, not a random sample, so proportions here do not describe all 27 losses.
- For each query the **loss driver** is the gold document whose nDCG contribution fell the most going from
  BM25 to hybrid.
- Groups are assigned by a fixed rule on where the *dense* retriever ranked that document (see
  `agentrag/bench/analysis.py`). The rule was fixed before reading any abstracts; the explanations below are
  my reading of the evidence and are labelled as such.

| Group | Rule (dense rank of the driver doc) | Queries | Distinct docs |
| --- | --- | --- | --- |
| Dense blind spot | > 50, or outside top 100 | 8 | 6 |
| Dense weak | 11–50 | 3 | 3 |
| Consensus demotion | ≤ 10 | 9 | 9 |

The 20 queries involve only **17 distinct gold documents**: one document drives three losses and another
drives two (the statin document, which appears in both Group 2 and Group 3), so these failures are correlated, not 20 independent events.

## Group 1: dense blind spot (8 queries)

The gold document is rank 1–9 for BM25 but outside the dense top 100 for 6 of the 8 queries, and at dense
rank 99 and 90 for the other two (690, 452). RRF then credits it almost entirely through BM25's list and it
falls behind documents both retrievers like.

What the abstracts show (I read these passages):

- **Venule/arteriole queries (1363, 128, 1362): one document.** The gold abstract is about fibroblast
  scaffolds and angiogenesis in infarcted heart tissue. "arterioles, venules" appears once, in a
  parenthetical in its final results sentence. BM25 ranks it 1st or 3rd on those two rare words;
  the dense model, which embeds the whole abstract, never surfaces it.
- **303 (DMRT1/MHM):** neither "DMRT1" nor "MHM" occurs in the gold abstract (chicken sex identity).
  BM25 reaches it through other shared terms; supporting the claim needs background knowledge.
- **324 (Raptor/G-CSF):** "Raptor" does not occur; "G-CSF" occurs three times (an mTOR/MDSC paper).
- **1179 (MDA5):** "MDA5" occurs once, in the opening sentence of a review. BM25 rank is only 9,
  so this is a weak signal to begin with.
- **690 (SFM/lactate):** the gold abstract is about volume depletion in Gabonese children with malaria.
  "lactate" and "Schimmelpenning" do not occur; the one rare-term overlap is "Gabonese". This looks like a
  lexical coincidence more than topical match (interpretation).
- **452:** the one case where hybrid (0.218) is worse than **both** BM25 (0.307) and dense alone (0.613).
  Dense ranks one gold document first, BM25 does not retrieve it at all, and BM25 ranks a second gold
  document third that dense ranks 90th. Fusion demotes both.

**Interpretation.** In the abstracts I read, BM25's win rests on a few rare tokens that are incidental
(venules, MDA5, Gabonese) or on general term overlap when the claim's key entity is absent from the
abstract (DMRT1, Raptor).
I cannot tell from rank data alone whether the encoder is deficient or BM25 was lucky.

**Hypothesis tested and rejected: input truncation.** bge-small truncates at 512 tokens, so matching terms
might sit in dropped text. Only 1 of the 20 driver documents exceeds 512 tokens (550, for query 674, which
is a consensus-demotion case, not a blind spot); the next longest is 421. Truncation does not explain these misses.

## Group 2: dense weak (3 queries)

Dense ranks the driver document 14th, 21st and 44th (queries 1100, 314, 619). It still reaches hybrid
ranks 6, 5 and 5, so the loss is a partial-credit demotion of the same kind as Group 3, with a weaker dense
signal. The losses are moderate (−0.11 to −0.28 nDCG@10).

## Group 3: consensus demotion (9 queries)

Dense also ranks the gold document in its top 10 but, in these queries, BM25's #1 or #2 hit slips to hybrid
rank 2–4.

**Mechanism, measured.** 16 documents overtook a gold document in these 9 queries (ahead in hybrid, behind
in BM25). **All 16 are in the top 5 of both retrievers.** RRF rewards agreement: a document ranked 2nd and
2nd scores 1/62 + 1/62 = 0.0323; a gold document ranked 1st by BM25 and 8th by dense scores
1/61 + 1/68 = 0.0311, so it loses. This is RRF behaving as designed.

**Why the agreed-on documents are not the gold ones (evidence, then interpretation).** Titles of overtakers
are topically close to the claim but not the labelled evidence. For query 845 (NETs released by
ANCA-stimulated neutrophils), the overtakers are general NET papers ("Neutrophil extracellular traps: Is
immunity the second function of chromatin?"; "Viable neutrophils release mitochondrial DNA to form
neutrophil extracellular traps") while the gold paper is "Netting neutrophils in autoimmune small-vessel
vasculitis". SciFact labels evidence for a *specific claim*, and topic-level similarity from both retrievers
points to neighbours. I did not verify that every overtaker lacks relevant content, since SciFact judgments
are sparse and an overtaker could be genuinely relevant but unlabelled.

## A cross-cutting observation: retrieval cannot see polarity

Two pairs of **mirror claims** retrieve the same document. "Arterioles have a larger lumen diameter than
venules" and "Venules have a larger lumen diameter than arterioles" (128, 1362) share a gold document, as do
"Statins increase blood cholesterol" and "Statins decrease blood cholesterol" (1100, 1099). Evidently
neither BM25 nor the embeddings distinguish direction. Telling support from contradiction needs a
verification step after retrieval (e.g. an NLI model). AgentRAG's current critic does not do this: it
checks lexical overlap with the context and citation validity only.

## Caveats

- 20 worst-case queries, 17 distinct documents, rule-based groups: suggestive, not statistical.
- Explanations in "interpretation" sentences come from reading five abstracts in detail (the
  Group 1 documents) plus titles of the overtaking documents; they are not established by an ablation.
- Not tested: weighted RRF, other fusion constants, or score-based fusion. They are natural next
  experiments for Group 3 and have not been run.
