#!/usr/bin/env python3
"""Run the SciFact claim-verification experiment.

Stages
  tune-nli   pick the NLI critic's confidence threshold on the *train* split (no API calls)
  run        retrieve, judge, critique and retry; evaluate every arm

Arms
  B0  Gemini judge, no critic
  B1  judge + rules + blind LLM polarity check          (the main method)
  B2  judge + rules + blind NLI polarity check          (non-LLM critic)
  B3  judge + rules only                                (ablation: what the polarity check adds)
  A   NLI model alone, no LLM                           (reference)
  oracle  judge given the gold abstracts, no critic     (separates retrieval from judging errors)

Safety rails
  * ``--split train`` uses a seeded 60-claim stratified subset for prompt development.
  * ``--split dev`` (the 300 BEIR test claims) refuses to run unless docs/results/verify_freeze.json
    exists with a fingerprint matching the current prompts and settings, and refuses to overwrite
    an earlier dev result: the final run happens once, with a configuration committed beforehand.
  * every response is cached; ``--offline`` makes any uncached request an error.

Usage
  python scripts/run_verification.py --stage tune-nli
  python scripts/run_verification.py --stage run --split train --arms B0,B1,B2,B3,A --label v1-low
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from agentrag.bench import data as beir_data
from agentrag.bench.registry import CACHE, RESULTS, make_getter
from agentrag.bench.stats import paired_bootstrap_metric
from agentrag.config import settings
from agentrag.verify import data as sf_data
from agentrag.verify.cache import ResponseCache, find_secrets
from agentrag.verify.critic import Critic, LlmPolarityChecker
from agentrag.verify.experiment import (
    claim_record,
    docs_for,
    evaluate_arm,
    nli_only_verdict,
    oracle_docs,
    tune_nli_only_tau,
    tune_tau,
    validation_items,
)
from agentrag.verify.gemini import THINKING_LEVELS, BudgetExceeded, GeminiClient, GenaiTransport
from agentrag.verify.graph import build_verify_graph, verify_claim
from agentrag.verify.judge import Judge
from agentrag.verify.metrics import checker_validation, contradiction_recall, macro_f1
from agentrag.verify.nli import NLI_MODEL, NliPolarityChecker, NliScorer
from agentrag.verify.prompts import JUDGE_VERSION, RELATION_VERSION, config_fingerprint
from agentrag.verify.ratelimit import DailyBudgetExhausted, RateLimiter, limits_from_env

CACHE_PATH = RESULTS / "verify_cache.jsonl"
LEDGER_PATH = Path(".verify_state/ledger.json")
NLI_TUNING = RESULTS / "verify_nli_tuning.json"
NLI_ONLY_TUNING = RESULTS / "verify_nli_only_tuning.json"
FREEZE_PATH = RESULTS / "verify_freeze.json"
MODEL = settings.gemini_model


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def retrieve(sf: sf_data.SciFact, claims, k: int) -> dict[str, list[str]]:
    """Top-k document ids per claim from the app's default hybrid (BM25 + bge-small, RRF)."""
    beir = beir_data.load("scifact", CACHE)
    sf_data.check_beir_parity(sf, beir.queries, beir.corpus)
    retriever = make_getter()("hybrid-bge")
    retriever.index(beir.corpus)
    results = retriever.search_batch([c.text for c in claims], k=k)
    return {c.id: [d for d, _ in hits] for c, hits in zip(claims, results, strict=True)}


def tune_nli_stage(sf: sf_data.SciFact) -> None:
    """Choose the NLI critic threshold using TRAIN-split human evidence only (no API calls)."""
    scorer = NliScorer()
    items = validation_items(sf, sf.train, seed=0)
    probs = scorer.score_many([(" ".join(i["sentences"]), i["claim"]) for i in items])
    from agentrag.verify.schemas import Relation

    samples = [(p, Relation(i["expected"])) for p, i in zip(probs, items, strict=True)]
    tuned = tune_tau(samples)
    tuned.update({
        "model": NLI_MODEL,
        "split": "train",
        "items": "gold rationale (entails/contradicts) + shuffled-claim negatives",
    })
    NLI_TUNING.write_text(json.dumps(tuned, indent=1), encoding="utf-8")
    print(f"tau = {tuned['tau']}  macro-F1 on {tuned['n']} train items = {tuned['macro_f1']:.3f}")
    print("grid:", {k: round(v, 3) for k, v in tuned["grid"].items()})


def build_client(args, offline: bool) -> tuple[GeminiClient, ResponseCache]:
    cache = ResponseCache(CACHE_PATH, offline=offline)
    limits = limits_from_env()
    if args.max_calls is not None:
        limits = dataclasses.replace(limits, max_calls=args.max_calls)
    limiter = RateLimiter(limits, LEDGER_PATH)
    transport = None if offline else GenaiTransport(MODEL, settings.google_api_key or "")
    client = GeminiClient(MODEL, cache, limiter, transport, thinking=args.thinking, max_usd=args.max_usd)
    return client, cache


def current_config(args) -> dict:
    return {
        "model": MODEL, "thinking": args.thinking, "temperature": 0.0, "k": args.k,
        "max_attempts": args.max_attempts, "judge_version": args.judge_version,
        "relation_version": args.relation_version, "nli_model": NLI_MODEL,
        "nli_tau": load_json(NLI_TUNING)["tau"] if NLI_TUNING.exists() else None,
        "failure_policy_reported": ["keep", "abstain"], "primary_failure_policy": "keep",
        "retrieval": "hybrid-bge (BM25 + bge-small-en-v1.5, RRF k=60, depth 100)",
    }


def check_freeze(config: dict, fingerprint: str) -> None:
    if not FREEZE_PATH.exists():
        sys.exit(f"refusing to run on dev: {FREEZE_PATH} does not exist (freeze the configuration first)")
    frozen = load_json(FREEZE_PATH)
    if frozen["fingerprint"] != fingerprint:
        sys.exit(f"refusing to run on dev: configuration fingerprint {fingerprint} != frozen {frozen['fingerprint']}")


def compare(claims, outcomes_by_arm) -> dict:
    gold = [c.label for c in claims]
    out: dict = {}
    for a, b in (("B1", "B0"), ("B2", "B0"), ("B3", "B0"), ("B1", "B2")):
        if a not in outcomes_by_arm or b not in outcomes_by_arm:
            continue
        for policy, getter in (("keep", lambda o: o.final_keep.value), ("abstain", lambda o: o.final_abstain.value)):
            pa = [getter(o) for o in outcomes_by_arm[a]]
            pb = [getter(o) for o in outcomes_by_arm[b]]
            for name, metric in (("macro_f1", macro_f1), ("contradiction_recall", contradiction_recall)):
                out[f"{a} - {b} | {policy} | {name}"] = paired_bootstrap_metric(gold, pa, pb, metric, n_boot=5000)
    return out


def validate_checkers(args, sf, claims, client, nli_checker) -> dict:
    items = validation_items(sf, claims, seed=0)
    gold = [i for i in items if i["kind"] == "gold"][: args.validation_gold]
    shuffled = [i for i in items if i["kind"] == "shuffled"][: args.validation_shuffled]
    chosen = gold + shuffled
    result: dict = {"items": {"gold": len(gold), "shuffled": len(shuffled)}}
    checkers = {}
    if chosen:
        checkers["llm"] = LlmPolarityChecker(client, args.relation_version)
    if nli_checker is not None:
        checkers["nli"] = nli_checker
    for name, checker in checkers.items():
        predicted = [checker.relation(i["claim"], i["sentences"]).relation.value for i in chosen]
        gold_pred, shuffled_pred = predicted[: len(gold)], predicted[len(gold):]
        result[name] = {
            "gold": checker_validation([i["expected"] for i in gold], gold_pred) if gold else None,
            "shuffled": checker_validation([i["expected"] for i in shuffled], shuffled_pred) if shuffled else None,
        }
    return result


def run_stage(args) -> int:
    sf = sf_data.load(CACHE)
    claims = sf.dev if args.split == "dev" else sf_data.stratified_subset(sf.train, args.per_label, args.seed)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    fingerprint = config_fingerprint(args.judge_version, args.relation_version, **{
        k: v for k, v in current_config(args).items() if k not in ("judge_version", "relation_version")})
    config = {**current_config(args), "fingerprint": fingerprint}
    out_path = RESULTS / f"verify_{args.split}_{args.label}.json"
    if args.split == "dev":
        check_freeze(config, fingerprint)
        if out_path.exists() and not args.offline:
            sys.exit(f"refusing to overwrite {out_path}: the dev run happens once per frozen configuration")
    print(f"split={args.split} claims={len(claims)} arms={arms} fingerprint={fingerprint}\n"
          f"model={MODEL} thinking={args.thinking} judge={args.judge_version} relation={args.relation_version}")

    retrieved = retrieve(sf, claims, args.k)
    docs = {c.id: docs_for(sf, retrieved[c.id]) for c in claims}
    with_evidence = [c for c in claims if c.evidence_docs]
    ceiling = {
        f"evidence_doc_hit@{k}": sum(bool(c.evidence_docs & set(retrieved[c.id][:k])) for c in with_evidence)
        / max(1, len(with_evidence))
        for k in (1, args.k)
    }

    client, cache = build_client(args, args.offline)
    judge = Judge(client, args.judge_version)
    scorer, nli_checker, tau = None, None, None
    if {"B2", "A"} & set(arms) or args.validation_gold or args.validation_shuffled:
        if not NLI_TUNING.exists():
            sys.exit("run --stage tune-nli first")
        scorer = NliScorer()
        tau = load_json(NLI_TUNING)["tau"]
        nli_checker = NliPolarityChecker(scorer.score, tau)

    llm_checker = LlmPolarityChecker(client, args.relation_version)
    graphs = {
        "B0": build_verify_graph(judge, None, args.max_attempts),
        "B3": build_verify_graph(judge, Critic(None), args.max_attempts),
        "B1": build_verify_graph(judge, Critic(llm_checker), args.max_attempts),
    }
    if nli_checker is not None:
        graphs["B2"] = build_verify_graph(judge, Critic(nli_checker), args.max_attempts)

    results: dict = {"config": config, "retrieval": ceiling, "claims": [c.id for c in claims], "arms": {}}
    outcomes_by_arm: dict = {}
    started = time.time()
    try:
        for arm in [a for a in ("B0", "B3", "B1", "B2") if a in arms]:
            outcomes = []
            for n, claim in enumerate(claims, start=1):
                outcomes.append(verify_claim(graphs[arm], claim.text, docs[claim.id]))
                if n % 10 == 0:
                    print(f"  {arm}: {n}/{len(claims)} claims | api calls this run {client.api_calls} | "
                          f"ledger today {client.limiter.ledger.count_today()}", flush=True)
            outcomes_by_arm[arm] = outcomes
            results["arms"][arm] = {
                "metrics": evaluate_arm(claims, outcomes, docs),
                "claims": [claim_record(c, o, retrieved[c.id]) for c, o in zip(claims, outcomes, strict=True)],
            }
            m = results["arms"][arm]["metrics"]
            retry = m["retry_keep"]
            print(f"{arm}: macro-F1 {m['final_keep']['macro_f1']:.3f} (initial {m['initial']['macro_f1']:.3f}) "
                  f"contradiction recall {m['final_keep']['per_class']['CONTRADICT']['recall']:.3f} "
                  f"flagged {retry['flagged']} fixed {retry['fixed']} harmed {retry['harmed']} "
                  f"| api calls total {m['usage']['api_calls_total']}", flush=True)

        if "oracle" in arms:
            oracle_graph = build_verify_graph(judge, None, args.max_attempts)
            oracle_docs_by_id = {c.id: oracle_docs(sf, c) for c in claims}
            outcomes = [verify_claim(oracle_graph, c.text, oracle_docs_by_id[c.id]) for c in claims]
            results["arms"]["oracle"] = {
                "metrics": evaluate_arm(claims, outcomes, oracle_docs_by_id),
                "claims": [
                    claim_record(c, o, [d.id for d in oracle_docs_by_id[c.id]])
                    for c, o in zip(claims, outcomes, strict=True)
                ],
            }
            oracle_f1 = results["arms"]["oracle"]["metrics"]["final_keep"]["macro_f1"]
            print(f"oracle: macro-F1 {oracle_f1:.3f}")

        if "A" in arms and scorer is not None:
            per_claim = []
            for claim in claims:
                sentences = [s for d in docs[claim.id] for s in d.sentences]
                per_claim.append(scorer.score_many([(s, claim.text) for s in sentences]) if sentences else [])
            gold = [c.label for c in claims]
            if args.split == "train":
                tuned = tune_nli_only_tau(list(zip(per_claim, gold, strict=True)))
                NLI_ONLY_TUNING.write_text(json.dumps(tuned, indent=1), encoding="utf-8")
            tau_a = load_json(NLI_ONLY_TUNING)["tau"]
            from agentrag.verify.metrics import summarize

            pred = [nli_only_verdict(p, tau_a).value for p in per_claim]
            summary = summarize(gold, pred)
            results["arms"]["A"] = {"tau": tau_a, "metrics": {"final_keep": summary}}
            print(f"A (NLI only, tau={tau_a}): macro-F1 {summary['macro_f1']:.3f}")

        results["comparisons"] = compare(claims, outcomes_by_arm)
        if args.validation_gold or args.validation_shuffled or nli_checker:
            results["checker_validation"] = validate_checkers(args, sf, claims, client, nli_checker)
            print("checker validation:", json.dumps(results["checker_validation"])[:600])
    except (DailyBudgetExhausted, BudgetExceeded) as err:
        print(f"\nSTOPPED: {err}\nEverything completed so far is cached; rerun the same command to resume.")
        return 2

    results["spend"] = {"api_calls_this_run": client.api_calls, "paid_equivalent_usd_this_run": client.spent_usd,
                        "wall_seconds": round(time.time() - started, 1), "cache_entries": len(cache)}
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    leaks = find_secrets(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else []
    size_mb = CACHE_PATH.stat().st_size / 1e6 if CACHE_PATH.exists() else 0.0
    print(f"\nwrote {out_path} | api calls this run {client.api_calls} | cache {len(cache)} entries, {size_mb:.2f} MB, "
          f"secret patterns found: {leaks or 'none'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("tune-nli", "run"), default="run")
    p.add_argument("--split", choices=("train", "dev"), default="train")
    p.add_argument("--arms", default="B0,B1,B2,B3")
    p.add_argument("--label", default="run")
    p.add_argument("--per-label", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--judge-version", default=JUDGE_VERSION)
    p.add_argument("--relation-version", default=RELATION_VERSION)
    p.add_argument("--thinking", choices=THINKING_LEVELS, default="low")
    p.add_argument("--offline", action="store_true", help="cache only; any uncached request is an error")
    p.add_argument("--max-calls", type=int, default=None, help="hard cap on API requests for this run")
    p.add_argument("--max-usd", type=float, default=None, help="stop at this paid-equivalent spend")
    p.add_argument("--validation-gold", type=int, default=0, help="LLM-checker items from gold rationales")
    p.add_argument("--validation-shuffled", type=int, default=0, help="LLM-checker shuffled-claim negative controls")
    args = p.parse_args()

    sf = sf_data.load(CACHE)
    if args.stage == "tune-nli":
        tune_nli_stage(sf)
        return 0
    return run_stage(args)


if __name__ == "__main__":
    sys.exit(main())
