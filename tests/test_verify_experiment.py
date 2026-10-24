import pytest

from agentrag.verify.common import RelationResult, Usage
from agentrag.verify.critic import Critic
from agentrag.verify.data import Claim, Doc, SciFact
from agentrag.verify.experiment import (
    TAU_GRID,
    claim_record,
    evaluate_arm,
    gold_pairs,
    nli_only_verdict,
    oracle_docs,
    predicted_pairs,
    tune_nli_only_tau,
    tune_tau,
    usage_summary,
    validation_items,
)
from agentrag.verify.graph import build_verify_graph, verify_claim
from agentrag.verify.judge import JudgeResult
from agentrag.verify.schemas import Citation, JudgeOutput, Relation, Verdict

D1 = Doc("1", "T1", ("Alpha rises.", "Beta falls.", "Gamma."))
D2 = Doc("2", "T2", ("Delta.", "Epsilon."))
D3 = Doc("3", "T3", ("Zeta.",))


def _sf():
    claims = [
        Claim("10", "Alpha rises.", "SUPPORT", {"1": {0}}, ("1",)),
        Claim("11", "Beta rises.", "CONTRADICT", {"1": {1}}, ("1",)),
        Claim("12", "Delta is odd.", "NOT_ENOUGH_INFO", {}, ("2",)),
        Claim("13", "Epsilon holds.", "SUPPORT", {"2": {1}}, ("2",)),
    ]
    return SciFact(docs={d.id: d for d in (D1, D2, D3)}, train=[], dev=claims), claims


class FakeJudge:
    def __init__(self, *outputs):
        self.outputs = list(outputs)

    def judge(self, claim, docs, problem=None):
        return JudgeResult(self.outputs.pop(0), "ok", Usage(api_calls=1, model_calls=1, prompt_tokens=100,
                                                              output_tokens=10, latency_ms=50.0, cost_usd=0.001))


class FakeChecker:
    name = "fake"

    def __init__(self, *relations):
        self.relations = list(relations)

    def relation(self, claim, sentences):
        return RelationResult(self.relations.pop(0), Usage(model_calls=1))


def _out(verdict, doc=None, ids=()):
    return JudgeOutput(verdict=verdict, citations=[Citation(doc_id=doc, sentence_ids=list(ids))] if doc else [])


def test_oracle_docs_use_evidence_docs_or_the_cited_docs_for_not_enough_info():
    sf, claims = _sf()
    assert [d.id for d in oracle_docs(sf, claims[0])] == ["1"]
    assert [d.id for d in oracle_docs(sf, claims[2])] == ["2"]


def test_gold_and_predicted_pairs():
    sf, claims = _sf()
    assert gold_pairs(claims[0]) == {("1", 0)} and gold_pairs(claims[2]) == set()
    outcome = verify_claim(build_verify_graph(FakeJudge(_out(Verdict.SUPPORT, "1", [0, 5])), None), "c", [D1])
    assert predicted_pairs(outcome, [D1]) == {("1", 0)}  # the out-of-range index is not a valid citation


def test_evaluate_arm_reports_verdict_retry_faithfulness_and_evidence_numbers():
    sf, claims = _sf()
    docs = {c.id: [D1, D2] for c in claims}
    judge = FakeJudge(
        _out(Verdict.SUPPORT, "1", [0]),            # claim 10: correct, passes
        _out(Verdict.SUPPORT, "1", [1]),            # claim 11: wrong (gold CONTRADICT), critic flags it...
        _out(Verdict.CONTRADICT, "1", [1]),         # ...and the retry fixes it
        _out(Verdict.NOT_ENOUGH_INFO),              # claim 12: correct NEI
        _out(Verdict.SUPPORT, "2", [1]),            # claim 13: correct, passes
    )
    checker = FakeChecker(Relation.ENTAILS, Relation.CONTRADICTS, Relation.CONTRADICTS, Relation.ENTAILS)
    graph = build_verify_graph(judge, Critic(checker))
    outcomes = [verify_claim(graph, c.text, docs[c.id]) for c in claims]

    result = evaluate_arm(claims, outcomes, docs)
    assert result["initial"]["accuracy"] == pytest.approx(3 / 4)
    assert result["final_keep"]["accuracy"] == 1.0 and result["final_keep"]["macro_f1"] == 1.0
    assert result["retry_keep"]["flagged"] == 1 and result["retry_keep"]["fixed"] == 1
    assert result["faithfulness_initial"] == {"faithful": 2, "judged": 3, "rate": pytest.approx(2 / 3)}
    assert result["faithfulness_final"]["rate"] == 1.0
    assert result["evidence"]["precision"] == 1.0 and result["evidence"]["recall"] == 1.0
    assert result["usage"]["api_calls_total"] == 5 and result["usage"]["attempts_mean"] == pytest.approx(5 / 4)
    assert result["first_attempt_status"] == {"ok": 4}


def test_usage_summary_averages_tokens_cost_and_latency_percentiles():
    outcomes = [verify_claim(build_verify_graph(FakeJudge(_out(Verdict.NOT_ENOUGH_INFO)), None), "c", [D1])
                for _ in range(4)]
    summary = usage_summary(outcomes)
    assert summary["prompt_tokens_per_claim"] == 100 and summary["paid_equivalent_usd_total"] == pytest.approx(0.004)
    assert summary["api_latency_ms_per_claim_p50"] == 50.0


def test_claim_record_is_json_serialisable_and_keeps_the_critics_reasoning():
    import json

    sf, claims = _sf()
    graph = build_verify_graph(FakeJudge(_out(Verdict.SUPPORT, "1", [0]), _out(Verdict.SUPPORT, "1", [0])),
                               Critic(FakeChecker(Relation.CONTRADICTS, Relation.ENTAILS)))
    record = claim_record(claims[0], verify_claim(graph, claims[0].text, [D1]), ["1", "2"])
    json.dumps(record)
    assert record["flagged"] and record["attempts"][0]["critic"]["relation"] == "contradicts"
    assert record["attempts"][1]["critic"]["passed"]


def test_nli_only_verdict_uses_the_strongest_sentence():
    weak = {"entailment": 0.2, "neutral": 0.7, "contradiction": 0.1}
    strong_c = {"entailment": 0.05, "neutral": 0.05, "contradiction": 0.9}
    strong_e = {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}
    assert nli_only_verdict([], 0.5) is Verdict.NOT_ENOUGH_INFO
    assert nli_only_verdict([weak, weak], 0.5) is Verdict.NOT_ENOUGH_INFO
    assert nli_only_verdict([weak, strong_c], 0.8) is Verdict.CONTRADICT
    assert nli_only_verdict([strong_e, strong_c], 0.8) is Verdict.NOT_ENOUGH_INFO  # equally strong conflict: abstain
    assert nli_only_verdict([strong_e, weak], 0.8) is Verdict.SUPPORT
    assert nli_only_verdict([strong_e], 0.95) is Verdict.NOT_ENOUGH_INFO  # below tau


def test_tau_tuning_picks_the_threshold_that_separates_the_classes_and_prefers_higher_on_ties():
    confident = {"entailment": 0.97, "neutral": 0.02, "contradiction": 0.01}
    unsure = {"entailment": 0.62, "neutral": 0.30, "contradiction": 0.08}
    samples = [(confident, Relation.ENTAILS)] * 5 + [(unsure, Relation.NEUTRAL)] * 5
    best = tune_tau(samples)
    # the entailment threshold can be anything in (0.62, 0.97]; the strictest wins, and the contradiction
    # threshold, which no sample exercises, ties at the strictest value too
    assert best["tau_entail"] == 0.95 and best["macro_f1"] > 0.6
    assert best["n"] == 10 and "0.5,0.5" in best["grid"] and best["per_class_recall"]["entails"]["recall"] == 1.0


def test_nli_only_tau_tuning_runs_on_sentence_level_samples():
    e = {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}
    n = {"entailment": 0.1, "neutral": 0.8, "contradiction": 0.1}
    result = tune_nli_only_tau([([e, n], "SUPPORT"), ([n], "NOT_ENOUGH_INFO")])
    assert result["macro_f1"] > 0 and result["tau"] in TAU_GRID


def test_validation_items_pair_gold_rationales_with_the_claim_and_with_a_different_claim():
    sf, claims = _sf()
    items = validation_items(sf, claims, seed=1)
    gold = [i for i in items if i["kind"] == "gold"]
    shuffled = [i for i in items if i["kind"] == "shuffled"]
    assert len(gold) == len(shuffled) == 3  # the NOT_ENOUGH_INFO claim has no rationale
    by_id = {i["id"]: i for i in gold}
    assert by_id["10"]["expected"] == "entails" and by_id["11"]["expected"] == "contradicts"
    assert by_id["10"]["sentences"] == ["Alpha rises."]
    for item in shuffled:
        original = next(c for c in claims if c.id == item["id"])
        assert item["expected"] == "neutral" and item["claim"] != original.text
        assert item["sentences"] == next(g["sentences"] for g in gold if g["id"] == item["id"])
    assert items == validation_items(sf, claims, seed=1)  # deterministic
