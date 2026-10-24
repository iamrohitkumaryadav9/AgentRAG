#!/usr/bin/env python3
"""Run the SciFact claim-verification experiment.

Stages
  tune-nli   pick the NLI critic's confidence thresholds on the *train* split (no API calls)
  run        retrieve, judge, critique and retry; evaluate every arm
  freeze     record the configuration the final dev run will use (docs/results/verify_freeze.json)

Arms
  B0  Gemini judge, no critic
  B3  judge + citation rules only
  N   judge + rules + NEI recheck                       (the NEI recheck alone)
  B1p judge + rules + blind LLM polarity check          (polarity check alone)
  B2p judge + rules + blind NLI polarity check          (non-LLM polarity check alone)
  B1  judge + rules + LLM polarity check + NEI recheck  (the main method)
  B2  judge + rules + NLI polarity check + NEI recheck  (the non-LLM critic)
  A   NLI model alone, no LLM                           (reference)
  oracle  judge given the gold abstracts, no critic     (separates retrieval from judging errors)

Safety rails
  * Budget, enforced in code (agentrag/verify/budget.py): all API calls stop at Rs 600 of tracked spend or 2,000
    requests, whichever comes first. Spend is computed from token counts and persisted in a per-key ledger.
  * Tripwire: a call costing more than 2x the ~$0.002 estimate, or any thinking token, stops every worker.
  * Responses are cached before anything else happens; a cached prompt is never requested again.
  * ``--split train`` uses a seeded 60-claim stratified subset for prompt development.
  * ``--split dev`` (the 300 BEIR test claims) refuses to run unless docs/results/verify_freeze.json exists with a
    fingerprint matching the current prompts and settings, and refuses to overwrite an earlier dev result.

Usage
  python scripts/run_verification.py --stage tune-nli
  python scripts/run_verification.py --stage run --split train --arms B0 --label v1-low --workers 6
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from agentrag.bench import data as beir_data
from agentrag.bench.registry import CACHE, RESULTS, make_getter
from agentrag.bench.stats import paired_bootstrap_metric
from agentrag.config import settings
from agentrag.verify import data as sf_data
from agentrag.verify.budget import Budget, BudgetExceeded, TripwireError, budget_from_env
from agentrag.verify.cache import ResponseCache, find_secrets
from agentrag.verify.critic import Critic, LlmPolarityChecker
from agentrag.verify.experiment import (
    NEI_TAU_GRID,
    claim_record,
    docs_for,
    evaluate_arm,
    nli_only_verdict,
    oracle_docs,
    simulate_nei_tau,
    tune_nli_only_tau,
    tune_tau,
    validation_items,
)
from agentrag.verify.gemini import THINKING_LEVELS, GeminiClient, GenaiTransport, Halted
from agentrag.verify.graph import build_verify_graph, verify_claim
from agentrag.verify.judge import Judge
from agentrag.verify.metrics import checker_validation, contradiction_recall, macro_f1
from agentrag.verify.nei_rule import NeiRecheck
from agentrag.verify.nli import NLI_MODEL, NliPolarityChecker, NliScorer
from agentrag.verify.prompts import JUDGE_VERSION, RELATION_VERSION, config_fingerprint
from agentrag.verify.ratelimit import (
    DailyBudgetExhausted,
    RateLimiter,
    ledger_path_for_key,
    limits_from_env,
)

CACHE_PATH = RESULTS / "verify_cache.jsonl"
NLI_TUNING = RESULTS / "verify_nli_tuning.json"
NLI_ONLY_TUNING = RESULTS / "verify_nli_only_tuning.json"
NEI_TUNING = RESULTS / "verify_nei_tuning.json"
FREEZE_PATH = RESULTS / "verify_freeze.json"
MODEL = settings.gemini_model
STOPPING = (DailyBudgetExhausted, BudgetExceeded, TripwireError, Halted)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def nli_taus() -> tuple[float, float]:
    tuned = load_json(NLI_TUNING)
    return tuned["tau_entail"], tuned["tau_contradict"]


def nei_tau_value(args) -> float | None:
    """The NEI-recheck threshold: --nei-tau if given, else the value tuned on train (verify_nei_tuning.json)."""
    if args.nei_tau is not None:
        return args.nei_tau
    return load_json(NEI_TUNING)["tau"] if NEI_TUNING.exists() else None


def retrieve(sf: sf_data.SciFact, claims, k: int) -> dict[str, list[str]]:
    """Top-k document ids per claim from the app's default hybrid (BM25 + bge-small, RRF)."""
    beir = beir_data.load("scifact", CACHE)
    sf_data.check_beir_parity(sf, beir.queries, beir.corpus)
    retriever = make_getter()("hybrid-bge")
    retriever.index(beir.corpus)
    results = retriever.search_batch([c.text for c in claims], k=k)
    return {c.id: [d for d, _ in hits] for c, hits in zip(claims, results, strict=True)}


def tune_nli_stage(sf: sf_data.SciFact) -> None:
    """Choose the NLI critic thresholds using TRAIN-split human evidence only (no API calls)."""
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
    print(f"tau_entail = {tuned['tau_entail']}  tau_contradict = {tuned['tau_contradict']}  "
          f"macro-F1 on {tuned['n']} train items = {tuned['macro_f1']:.3f}")
    print("per-class recall:", {k: round(v['recall'], 3) for k, v in tuned["per_class_recall"].items()})


def tune_nei_stage(args) -> int:
    """Choose the NEI-recheck threshold from retries run at the lowest threshold (no API calls).

    Reads arm N of a train run made with a permissive --nei-tau, evaluates every stricter threshold on the stored
    retry outcomes, and picks the best net (fixes - harms) subject to harms <= fixes. Train claims only.
    """
    source = RESULTS / f"verify_train_{args.label}.json"
    run = load_json(source)
    if "N" not in run["arms"]:
        sys.exit(f"{source} has no arm N (rules + NEI recheck); run it first with --arms ...,N --nei-tau 0.5")
    low = run["config"]["nei_recheck_tau"]
    grid = [t for t in NEI_TAU_GRID if t >= low]
    result = simulate_nei_tau(run["arms"]["N"]["claims"], grid)
    result.update({"source": source.name, "lowest_threshold_run": low, "split": "train"})
    NEI_TUNING.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"NEI-recheck threshold: {result['tau']}  ({result['nei_verdicts']} NOT_ENOUGH_INFO verdicts, "
          f"{result['false_nei']} wrong / {result['correct_nei']} right)")
    for tau, row in result["table"].items():
        print(f"  tau {tau}: flagged {row['flagged']:>2}  fixed {row['fixed']:>2}  "
              f"harmed {row['harmed']:>2}  net {row['net']:+d}")
    return 0


def build_client(args, offline: bool) -> tuple[GeminiClient, ResponseCache, Budget]:
    cache = ResponseCache(CACHE_PATH, offline=offline)
    key = settings.google_api_key or ""
    limits = dataclasses.replace(limits_from_env(), rpm=args.rpm or limits_from_env().rpm, tpm=None)
    if args.max_calls is not None:
        limits = dataclasses.replace(limits, max_calls=args.max_calls)
    budget = budget_from_env()
    # one ledger per API key: a new key (project, quota, credits) never inherits another project's totals
    limiter = RateLimiter(limits, ledger_path_for_key(key) if key else None, budget=budget)
    transport = None if offline else GenaiTransport(MODEL, key)
    if args.phase:
        print("phase:", limiter.start_phase(args.phase, args.phase_cap_inr))
    client = GeminiClient(MODEL, cache, limiter, transport, thinking=args.thinking)
    return client, cache, budget


def current_config(args) -> dict:
    return {
        "model": MODEL, "thinking": args.thinking, "temperature": 0.0, "k": args.k,
        "max_attempts": args.max_attempts, "judge_version": args.judge_version,
        "relation_version": args.relation_version, "nli_model": NLI_MODEL,
        "nli_taus": nli_taus() if NLI_TUNING.exists() else None,
        "nli_only_tau": load_json(NLI_ONLY_TUNING)["tau"] if NLI_ONLY_TUNING.exists() else None,
        "nei_recheck_tau": nei_tau_value(args),
        "nei_recheck": "NLI over every retrieved sentence; flag a NOT_ENOUGH_INFO verdict if entail or contradict "
                       "probability > tau and retry once with feedback naming the sentence",
        "tripwire_usd_per_call": budget_from_env().tripwire_usd,
        "thinking_tokens": "logged and reported per arm; not a stop condition",
        "failure_policy_reported": ["keep", "abstain"], "primary_failure_policy": "keep",
        "retrieval": "hybrid-bge (BM25 + bge-small-en-v1.5, RRF k=60, depth 100)",
    }


def fingerprint_for(args) -> tuple[dict, str]:
    config = current_config(args)
    rest = {k: v for k, v in config.items() if k not in ("judge_version", "relation_version")}
    return config, config_fingerprint(args.judge_version, args.relation_version, **rest)


def check_freeze(fingerprint: str) -> None:
    if not FREEZE_PATH.exists():
        sys.exit(f"refusing to run on dev: {FREEZE_PATH} does not exist (freeze the configuration first)")
    frozen = load_json(FREEZE_PATH)
    if frozen["fingerprint"] != fingerprint:
        sys.exit(f"refusing to run on dev: configuration fingerprint {fingerprint} != frozen {frozen['fingerprint']}")


def run_parallel(label: str, items: Sequence, fn: Callable, workers: int, client: GeminiClient) -> list:
    """Map ``fn`` over ``items`` with a bounded worker pool, preserving order.

    The first failure (tripwire, budget, quota, anything) halts the client so no worker sends another
    request, then re-raises. Cached prompts are already on disk, so a stopped run resumes where it left off.
    """
    results: list = [None] * len(items)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        try:
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                done += 1
                if done % 10 == 0 or done == len(items):
                    spend = client.limiter.spend_summary()
                    print(f"  {label}: {done}/{len(items)} | requests this run {client.api_calls} | "
                          f"spend Rs {spend.get('lifetime_inr', 0):.2f}", flush=True)
        except BaseException as exc:
            client.halt(f"{type(exc).__name__}: {exc}")
            for future in futures:
                future.cancel()
            raise
    return results


def compare(claims, outcomes_by_arm) -> dict:
    gold = [c.label for c in claims]
    out: dict = {}
    # Pairs chosen so each component's effect is isolated: N vs B3 is the NEI recheck alone; B1 vs B1p and
    # B2 vs B2p are the recheck added to each polarity critic; the *p arms vs B0 are the polarity checks alone.
    pairs = (("B3", "B0"), ("N", "B3"), ("B1p", "B0"), ("B2p", "B0"), ("B1", "B1p"), ("B2", "B2p"),
             ("B1", "B0"), ("B2", "B0"), ("B1", "B2"))
    for a, b in pairs:
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
    all_gold = [i for i in items if i["kind"] == "gold"]
    all_shuffled = [i for i in items if i["kind"] == "shuffled"]
    # The LLM checker costs API calls, so it sees only the requested number of items; the local NLI model sees all.
    subsets = {
        "llm": (all_gold[: args.validation_gold], all_shuffled[: args.validation_shuffled]),
        "nli": (all_gold, all_shuffled),
    }
    checkers = {}
    if args.validation_gold or args.validation_shuffled:
        checkers["llm"] = LlmPolarityChecker(client, args.relation_version)
    if nli_checker is not None:
        checkers["nli"] = nli_checker

    result: dict = {}
    for name, checker in checkers.items():
        gold, shuffled = subsets[name]
        chosen = gold + shuffled

        def predict(item, checker=checker):
            return checker.relation(item["claim"], item["sentences"]).relation.value

        if name == "llm":
            predicted = run_parallel("checker-validation", chosen, predict, args.workers, client)
        else:
            predicted = [predict(i) for i in chosen]
        gold_pred, shuffled_pred = predicted[: len(gold)], predicted[len(gold):]
        result[name] = {
            "items": {"gold": len(gold), "shuffled": len(shuffled)},
            "gold": checker_validation([i["expected"] for i in gold], gold_pred) if gold else None,
            "shuffled": checker_validation([i["expected"] for i in shuffled], shuffled_pred) if shuffled else None,
        }
    return result


def spend_line(limiter: RateLimiter, budget: Budget) -> str:
    s = limiter.spend_summary()
    return (f"lifetime {s['lifetime_calls']} requests, ${s['lifetime_usd']:.4f} = Rs {s['lifetime_inr']:.2f} "
            f"(caps: Rs {budget.max_inr:.0f} / {budget.max_calls} requests)")


def run_stage(args) -> int:
    sf = sf_data.load(CACHE)
    claims = sf.dev if args.split == "dev" else sf_data.stratified_subset(sf.train, args.per_label, args.seed)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    config, fingerprint = fingerprint_for(args)
    config = {**config, "fingerprint": fingerprint}
    out_path = RESULTS / f"verify_{args.split}_{args.label}.json"
    if args.split == "dev":
        check_freeze(fingerprint)
        if out_path.exists() and not args.offline:
            sys.exit(f"refusing to overwrite {out_path}: the dev run happens once per frozen configuration")
    print(f"split={args.split} claims={len(claims)} arms={arms} workers={args.workers} fingerprint={fingerprint}\n"
          f"model={MODEL} thinking={args.thinking} judge={args.judge_version} relation={args.relation_version}")

    retrieved = retrieve(sf, claims, args.k)
    docs = {c.id: docs_for(sf, retrieved[c.id]) for c in claims}
    with_evidence = [c for c in claims if c.evidence_docs]
    ceiling = {
        f"evidence_doc_hit@{k}": sum(bool(c.evidence_docs & set(retrieved[c.id][:k])) for c in with_evidence)
        / max(1, len(with_evidence))
        for k in (1, args.k)
    }

    client, cache, budget = build_client(args, args.offline)
    print("budget:", spend_line(client.limiter, budget))
    judge = Judge(client, args.judge_version)
    needs_nli = bool({"A", "B2", "B2p", "N", "B1"} & set(arms)) or args.validation_gold or args.validation_shuffled
    scorer, nli_checker, nei_rule = None, None, None
    if needs_nli:
        if not NLI_TUNING.exists():
            sys.exit("run --stage tune-nli first")
        scorer = NliScorer()
        tau_e, tau_c = nli_taus()
        nli_lock = threading.Lock()  # the torch model is shared by all workers

        def locked_score(premise: str, hypothesis: str):
            with nli_lock:
                return scorer.score(premise, hypothesis)

        def locked_score_many(pairs):
            with nli_lock:
                return scorer.score_many(pairs)

        nli_checker = NliPolarityChecker(locked_score, tau_e, tau_c)
        nei_tau = nei_tau_value(args)
        if {"N", "B1", "B2"} & set(arms):
            if nei_tau is None:
                sys.exit("the NEI recheck needs a threshold: pass --nei-tau (0.5 to explore) or run --stage tune-nei")
            nei_rule = NeiRecheck(locked_score_many, nei_tau)

    llm_checker = LlmPolarityChecker(client, args.relation_version)
    # Critics per arm. B1/B2 = polarity check + NEI recheck; B1p/B2p = polarity check only; N = NEI recheck only
    # (no polarity check); B3 = citation rules only; B0 = no critic.
    critics = {
        "B0": None,
        "B3": Critic(None),
        "B1p": Critic(llm_checker),
        "N": Critic(None, nei_rule),
        "B1": Critic(llm_checker, nei_rule),
    }
    if nli_checker is not None:
        critics["B2p"] = Critic(nli_checker)
        critics["B2"] = Critic(nli_checker, nei_rule)
    graphs = {name: build_verify_graph(judge, critic, args.max_attempts) for name, critic in critics.items()}

    results: dict = {"config": config, "retrieval": ceiling, "claims": [c.id for c in claims], "arms": {}}
    outcomes_by_arm: dict = {}
    started = time.time()
    try:
        for arm in [a for a in ("B0", "B3", "N", "B1p", "B2p", "B1", "B2") if a in arms]:
            outcomes = run_parallel(
                arm, claims, lambda claim, arm=arm: verify_claim(graphs[arm], claim.text, docs[claim.id]),
                args.workers, client,
            )
            outcomes_by_arm[arm] = outcomes
            results["arms"][arm] = {
                "metrics": evaluate_arm(claims, outcomes, docs),
                "claims": [claim_record(c, o, retrieved[c.id]) for c, o in zip(claims, outcomes, strict=True)],
            }
            m = results["arms"][arm]["metrics"]
            retry, on_nei = m["retry_keep"], m["retry_nei_subset"] or {}
            print(f"{arm}: macro-F1 {m['final_keep']['macro_f1']:.3f} (initial {m['initial']['macro_f1']:.3f}) "
                  f"contradiction recall {m['final_keep']['per_class']['CONTRADICT']['recall']:.3f} "
                  f"flagged {retry['flagged']} fixed {retry['fixed']} harmed {retry['harmed']} "
                  f"| on NEI verdicts: flagged {on_nei.get('flagged', 0)} fixed {on_nei.get('fixed', 0)} "
                  f"harmed {on_nei.get('harmed', 0)} | api calls total {m['usage']['api_calls_total']}", flush=True)
            u = m["usage"]
            print(f"   cost per LLM call: max ${u['max_call_usd']:.4f} mean ${u['mean_call_usd']:.4f} | thinking: "
                  f"max {u['max_thought_tokens']} mean/call {u['mean_thought_tokens_per_call']:.1f} "
                  f"({u['calls_with_thinking']} calls used thinking)", flush=True)

        if "oracle" in arms:
            oracle_graph = build_verify_graph(judge, None, args.max_attempts)
            oracle_docs_by_id = {c.id: oracle_docs(sf, c) for c in claims}
            outcomes = run_parallel(
                "oracle", claims, lambda c: verify_claim(oracle_graph, c.text, oracle_docs_by_id[c.id]),
                args.workers, client,
            )
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
            print("checker validation:", json.dumps(results["checker_validation"])[:700])
    except STOPPING as err:
        reason = client.halt_reason or str(err)
        print(f"\nSTOPPED: {reason}\n{spend_line(client.limiter, budget)}\n"
              "Everything completed so far is cached; rerun the same command to resume.")
        return 2

    results["spend"] = {
        **client.limiter.spend_summary(),
        "api_requests_this_run": client.api_calls,
        "paid_equivalent_usd_this_run": client.spent_usd,
        "wall_seconds": round(time.time() - started, 1),
        "cache_entries": len(cache),
        "workers": args.workers,
    }
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    leaks = find_secrets(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else []
    size_mb = CACHE_PATH.stat().st_size / 1e6 if CACHE_PATH.exists() else 0.0
    print(f"\nwrote {out_path} | requests this run {client.api_calls} | {spend_line(client.limiter, budget)} | "
          f"cache {len(cache)} entries, {size_mb:.2f} MB, secret patterns found: {leaks or 'none'}")
    return 0


def freeze_stage(args) -> int:
    """Record the configuration the one-shot dev run must match. Writes docs/results/verify_freeze.json."""
    config, fingerprint = fingerprint_for(args)
    evidence = {}
    for label in args.evidence:
        path = RESULTS / f"verify_train_{label}.json"
        if path.exists():
            arm = load_json(path)["arms"].get("B0")
            if arm:
                evidence[label] = {
                    "judge_version": load_json(path)["config"]["judge_version"],
                    "b0_macro_f1": arm["metrics"]["initial"]["macro_f1"],
                    "b0_contradiction_recall": arm["metrics"]["initial"]["per_class"]["CONTRADICT"]["recall"],
                }
    nei = load_json(NEI_TUNING) if NEI_TUNING.exists() else None
    rule_kept = nei is not None and nei["tau"] is not None
    dev_arm_order = ["B0", "B3", "N", "B1p", "B2p", "B1", "B2", "A", "oracle"] if rule_kept else [
        "B0", "B3", "B1p", "B2p", "A", "oracle"]
    arms = list(dev_arm_order)
    train_final = RESULTS / f"verify_train_{args.train_final}.json"
    budget = budget_from_env()
    freeze = {
        "fingerprint": fingerprint,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "selection_rules": {
            "judge_prompt": "judge-v1 frozen by decision: judge-v2's macro-F1 edge was one claim and v2 was written "
                            "after seeing v1's errors on the same 60 train claims; v2 is a train-only ablation",
            "nei_recheck": "maximise (fixes - harms) over tau in 0.5/0.7/0.9/0.95/0.98 subject to fixes > harms; "
                           "if none qualifies the rule is dropped and B1/B2 are the polarity-only variants",
        },
        "nei_recheck_decision": {"kept": rule_kept, "tuning": nei},
        "pre_declared_analysis": {
            "primary": ["B1 - B0: macro-F1 (keep policy)", "B1 - B0: contradiction recall (keep policy)"],
            "secondary_isolating_components": [
                "N - B3 (the NEI recheck alone)", "B1p - B0 and B2p - B0 (polarity checks alone)",
                "B1 - B1p and B2 - B2p (the recheck added to each polarity check)", "B1 - B2 (LLM vs NLI critic)",
            ],
            "method": "paired bootstrap over claims, 5000 resamples, 95% percentile intervals; the abstain policy is "
                      "reported alongside. Secondary comparisons are exploratory and uncorrected.",
            "no_changes_after_freeze": "no prompt, rule, threshold or policy changes once the dev run starts; "
                                       "the dev run executes once",
        },
        "train_evidence": evidence,
        "train_final_result_file": train_final.name if train_final.exists() else None,
        "budget": {
            "global_cap_inr": budget.max_inr, "global_cap_requests": budget.max_calls,
            "dev_phase_cap_inr": 250, "per_call_tripwire_usd": budget.tripwire_usd,
            "thinking_tokens": "logged, not a stop condition",
        },
        "dev_arms": arms,
        "dev_run_command": (
            f"python scripts/run_verification.py --stage run --split dev --label frozen "
            f"--judge-version {args.judge_version} --relation-version {args.relation_version} "
            f"--thinking {args.thinking} --k {args.k} --max-attempts {args.max_attempts} "
            f"--arms {','.join(dev_arm_order)} "
            f"--workers {args.workers} --rpm {args.rpm} --validation-gold 188 --validation-shuffled 60 "
            "--phase dev --phase-cap-inr 250"
        ),
    }
    FREEZE_PATH.write_text(json.dumps(freeze, indent=1), encoding="utf-8")
    print(f"froze fingerprint {fingerprint}\n{json.dumps(freeze['config'], indent=1)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("tune-nli", "tune-nei", "run", "freeze"), default="run")
    p.add_argument("--split", choices=("train", "dev"), default="train")
    p.add_argument("--arms", default="B0,B1,B2,B3")
    p.add_argument("--label", default="run")
    p.add_argument("--per-label", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--nei-tau", type=float, default=None, help="NEI-recheck threshold (default: tuned value)")
    p.add_argument("--judge-version", default=JUDGE_VERSION)
    p.add_argument("--relation-version", default=RELATION_VERSION)
    p.add_argument("--thinking", choices=THINKING_LEVELS, default="low")
    p.add_argument("--workers", type=int, default=6, help="concurrent claims (1-8)")
    p.add_argument("--rpm", type=int, default=60, help="client-side requests per minute (paid-tier guess)")
    p.add_argument("--phase", default=None, help="name of the study phase whose spend is capped (ledger-backed)")
    p.add_argument("--phase-cap-inr", type=float, default=None, help="rupee cap for --phase, from its first use")
    p.add_argument("--offline", action="store_true", help="cache only; any uncached request is an error")
    p.add_argument("--max-calls", type=int, default=None, help="extra cap on API requests for this run")
    p.add_argument("--validation-gold", type=int, default=0, help="LLM-checker items from gold rationales")
    p.add_argument("--validation-shuffled", type=int, default=0, help="LLM-checker shuffled-claim negative controls")
    p.add_argument("--evidence", nargs="*", default=[], help="train result labels to record in the freeze file")
    p.add_argument("--train-final", default="v1-final", help="final train result label to cite in the freeze")
    args = p.parse_args()
    args.workers = max(1, min(8, args.workers))

    sf = sf_data.load(CACHE)
    if args.stage == "tune-nli":
        tune_nli_stage(sf)
        return 0
    if args.stage == "tune-nei":
        return tune_nei_stage(args)
    if args.stage == "freeze":
        return freeze_stage(args)
    return run_stage(args)


if __name__ == "__main__":
    sys.exit(main())
