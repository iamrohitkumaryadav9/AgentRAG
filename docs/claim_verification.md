# Claim verification on SciFact: design and train-subset development

**Status.** The pipeline, metrics, budget controls and tests are built, and the configuration was frozen after
development on a **60-claim stratified subset of the train split** (20 SUPPORT, 20 CONTRADICT, 20 NOT_ENOUGH_INFO).
The final run on the 300 dev claims (the BEIR test queries) has **not** been run. Every number below is a train-subset
development number: with 60 claims, one claim moves macro-F1 by about 0.015, and none of the differences here should
be read as established. The frozen configuration is in [results/verify_freeze.json](results/verify_freeze.json).

## What it does

For each claim: retrieve the top 5 abstracts with the app's default hybrid retriever (BM25 + bge-small, RRF), have
Gemini (`gemini-3.8-flash`, temperature 0, thinking `low`) return a verdict (SUPPORT, CONTRADICT or NOT_ENOUGH_INFO)
with the sentences it cites, then let a **critic** check the answer and, on failure, retry once with specific
feedback (LangGraph: judge → critic → accept | retry, at most two attempts). Labels and gold rationale sentences
come from the original SciFact release; the 300 BEIR test queries are exactly its 300 dev claims (checked on every
run: 300 of 300 ids and texts identical, same 5,183 documents).

The critic has three independent parts, which are run as separate **arms** so each one's effect can be isolated:

| Arm | Judge + … |
| --- | --- |
| B0 | nothing (no critic) |
| B3 | citation rules: every cited sentence must exist, and a decisive verdict must cite something |
| N | rules + **NEI recheck** (below) |
| B1p | rules + a *blind* LLM polarity check |
| B2p | rules + a blind NLI polarity check (DeBERTa-v3, no LLM) |
| B1 | rules + LLM polarity check + NEI recheck |
| B2 | rules + NLI polarity check + NEI recheck |

The polarity check is *blind*: it sees only the claim and the sentences the judge cited, never the verdict or
rationale, so it cannot simply agree with the judge. It reports whether those sentences entail, contradict or are
neutral to the claim, and the critic compares that with what the verdict asserts. **Faithfulness** is the share of
SUPPORT / CONTRADICT verdicts whose cited sentences entail the verdict according to this check. A separate arm A
(NLI alone, no LLM) and an oracle arm (the judge given the gold abstracts) are defined for the dev run.

## Train-subset results (60 claims, judge-v1, thinking `low`)

"Keep" is the primary failure policy (a still-rejected answer is kept); "abstain" falls back to NOT_ENOUGH_INFO.
*Fixed* means initially wrong and finally right; *harmed* means initially right and finally wrong.

| Arm | Macro-F1 (keep) | Macro-F1 (abstain) | CONTRADICT recall | SUPPORT recall | Flagged | Fixed | Harmed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 judge only | 0.853 | 0.853 | 0.85 | 0.70 | 0 | 0 | 0 |
| B3 + citation rules | 0.853 | 0.853 | 0.85 | 0.70 | 0 | 0 | 0 |
| **N + NEI recheck only** | **0.885** | 0.885 | 0.90 | 0.75 | 23 | 2 | 0 |
| B1p + LLM polarity | 0.836 | 0.836 | 0.85 | 0.65 | 1 | 0 | 1 |
| B2p + NLI polarity | 0.836 | 0.765 | 0.85 | 0.65 | 5 | 0 | 1 |
| B1 LLM polarity + NEI recheck | 0.868 | 0.851 | 0.90 | 0.70 | 24 | 2 | 1 |
| B2 NLI polarity + NEI recheck | 0.868 | 0.780 | 0.90 | 0.70 | 28 | 2 | 1 |

Fix and harm **rates** (fix rate = fixed ÷ claims that were wrong and flagged; harm rate = harmed ÷ claims that
were right and flagged): N 0.25 and 0.00; B1 0.25 and 0.06; B2 0.25 and 0.05. The polarity-only arms have no
fixes, so their fix rate is undefined.

What this does and does not show:

- **The judge's decisive verdicts were all correct on this subset** (precision 1.00 for SUPPORT and for
  CONTRADICT): all 9 of its errors were false NOT_ENOUGH_INFO verdicts. A polarity check has nothing to fix when the
  judge makes no polarity errors, and both polarity-only arms scored *below* the judge alone (−0.017, one harmed
  claim each). On the dev claims this may differ; the polarity checks are kept in the dev run for that reason,
  not because this subset supports them.
- **The only improvement comes from the NEI recheck:** +0.032 macro-F1 over rules alone (95% CI [0.000, 0.082]),
  from 2 fixed claims. That interval touches zero.
- **The LLM checker is a far better polarity judge than the NLI model**, measured against human-marked evidence
  (known-answer items built from gold rationale sentences): it recovers 95% of contradictions and 75% of
  entailments (n = 40, accuracy 0.85) and calls 100% of shuffled-claim negative controls neutral (n = 20). The NLI
  model recovers 85% and 35% (it is biased toward "neutral" on scientific text) and 85% of the controls.
  Faithfulness of the judge's decisive verdicts: 0.968 (30 of 31) by the LLM checker, 0.839 (26 of 31) by NLI,
  so the NLI figure mostly reflects the NLI model's own misses.
- Cited sentences versus the human rationale (40 evidence claims): precision 0.67, recall 0.51 (recall is
  depressed by the false NOT_ENOUGH_INFO verdicts, which cite nothing).

## The NOT_ENOUGH_INFO recheck rule

Every judge error on the train subset was a false NOT_ENOUGH_INFO that cites nothing, which neither the citation
rules nor the polarity check can see. The rule works from the other side: when the verdict is NOT_ENOUGH_INFO, an NLI
model scores every retrieved sentence against the claim; if one entails or contradicts it with probability above τ,
the verdict is flagged and the judge retries with feedback naming that sentence. The threshold was chosen on the
train subset by the pre-stated rule *maximise fixes − harms subject to fixes > harms; drop the rule if none
qualifies*, evaluated offline from the stored retry outcomes at the most permissive threshold:

| τ | Flagged | Fixed | Harmed | Net |
| --- | --- | --- | --- | --- |
| **0.5** | 23 | 2 | 0 | **+2** |
| 0.7 | 20 | 1 | 0 | +1 |
| 0.9 | 14 | 1 | 0 | +1 |
| 0.95 | 11 | 1 | 0 | +1 |
| 0.98 | 9 | 1 | 0 | +1 |

Of the judge's 29 NOT_ENOUGH_INFO verdicts, 9 were wrong and 20 right. The rule qualified at every threshold and
τ = 0.5 won, so it stays in the frozen pipeline. Read it carefully: it is a **weak discriminator**. At τ = 0.5 it
flagged 23 verdicts of which only 8 were wrong (precision 0.35), because topically close sentences score high even
for correctly unanswerable claims. It did no harm here only because the judge, asked to reconsider, kept every one of
the 15 correct NOT_ENOUGH_INFO answers that were flagged. The benefit is 2 claims of 60.

## Judge prompt ablation (train only)

A single prompt revision, judge-v2, was written after judge-v1's errors were seen, aiming at the false-NEI bias, and
compared on the same 60 claims:

| Judge | Macro-F1 | CONTRADICT recall | SUPPORT recall | False NOT_ENOUGH_INFO |
| --- | --- | --- | --- | --- |
| v1 | 0.853 | 0.85 | 0.70 | 9 |
| v2 | 0.868 | 0.90 | 0.70 | 8 |

**A prompt revision did not fix the NEI bias; in 6 of the 8 remaining errors the gold evidence was retrieved at
rank 1–2.** v2 fixed one claim, an edge too small to count, and because it was written after seeing v1's errors on
these same claims even that is optimistic. judge-v1 is therefore the frozen judge. v2 used thinking tokens on one of
its 60 calls (219 tokens), which is what first showed that the model's lowest thinking level is not "off".

## Cost, latency and thinking

`gemini-3.8-flash` cannot disable thinking (`thinking_budget=0` is accepted but ignored; `MINIMAL` is rejected), so
thinking is set to `low` and **logged, not limited**. Per LLM call on the train subset (judge-v1):

| Arm | LLM calls | Cost per call, max / mean | Thinking tokens, max / mean per call | Cost per claim |
| --- | --- | --- | --- | --- |
| B0 | 60 | $0.0031 / $0.0019 | 0 / 0 | $0.0019 |
| N | 83 | $0.0057 / $0.0021 | 1,060 / 47 | $0.0029 |
| B1p | 92 | $0.0031 / $0.0014 | 410 / 15 | $0.0021 |
| B2p | 65 | $0.0032 / $0.0020 | 311 / 5 | $0.0021 |
| B1 | 117 | $0.0057 / $0.0016 | 1,060 / 45 | $0.0031 |
| B2 | 88 | $0.0057 / $0.0021 | 1,060 / 48 | $0.0031 |

First-pass judge calls never used thinking; the retries that ask the model to reconsider occasionally do (the
largest, 1,060 tokens, added about Rs 0.35 to that call). Typical API latency is 2–3 s per call
once the service is not congested (earlier calls took up to 100 s). Costs are the paid-equivalent list price
($0.75 / $3.75 per million input / output tokens, thinking billed as output); the Rs figures use 600/6.8 rupees per
dollar.

## Budget and operational notes

Spending limits are enforced in code (`agentrag/verify/budget.py`), not by intention: all API calls stop at Rs 600
of tracked spend or 2,000 requests, spend is persisted per API key, every in-flight request reserves its worst-case
cost before it is sent, a single call costing more than $0.02 stops everything, and each phase of the study has its
own rupee cap. Every response is cached, and a cached prompt is never sent again. Incidents worth knowing about: a
free-tier project allowed only 20 requests per day and stopped the first attempt (the client recognised the quota and
stopped cleanly); a Windows file-lock error while saving the spend ledger once discarded one paid response before it
was cached (one prompt was paid for twice, fixed with a regression test); and a too-tight cost tripwire halted the
run when retries began thinking (tripwire since set to $0.02).

## What happens next, and what could embarrass this result

The dev run executes once, with nothing changed after the freeze. Pre-declared primary comparisons are B1 − B0 on
macro-F1 and on contradiction recall, with paired-bootstrap 95% intervals; the component comparisons (N vs B3, the
polarity-only arms vs B0, B1 vs B1p, B1 vs B2) are exploratory and uncorrected. Expect the intervals to be wide:
with 64 CONTRADICT claims, differences under about 0.05 macro-F1 cannot be resolved.

Risks stated in advance: the train subset's 2-claim NEI gain may not replicate; the polarity checks may again do
nothing or harm; Gemini may have seen SciFact during pretraining; "NOT_ENOUGH_INFO" in SciFact means the cited
abstract lacked evidence, not that none exists in the corpus, so some "false" SUPPORT / CONTRADICT verdicts on those
claims may be defensible; and the critic cannot recover evidence the judge never cites unless the NEI recheck finds
the sentence.
