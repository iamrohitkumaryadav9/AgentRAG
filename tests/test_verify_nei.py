import pytest

from agentrag.verify.common import RelationResult, Usage
from agentrag.verify.critic import Critic
from agentrag.verify.data import Doc
from agentrag.verify.experiment import _nei_subset_stats, simulate_nei_tau
from agentrag.verify.graph import build_verify_graph, verify_claim
from agentrag.verify.judge import JudgeResult
from agentrag.verify.nei_rule import NeiRecheck
from agentrag.verify.schemas import Citation, JudgeOutput, Relation, Verdict

DOCS = [
    Doc("7", "Vessels", ("Arterioles are thick.", "Venules have a thin layer.", "Capillaries are tiny.")),
    Doc("9", "Other", ("Unrelated sentence.",)),
]
CLAIM = "Venules have a thinner layer than arterioles."


def probs(entail=0.0, contra=0.0):
    return {"entailment": entail, "neutral": 1 - entail - contra, "contradiction": contra}


class FakeScorer:
    """Scores sentences by a lookup on their text; counts calls."""

    def __init__(self, table):
        self.table, self.calls = table, 0

    def __call__(self, pairs):
        self.calls += 1
        return [self.table.get(sentence, probs()) for sentence, _ in pairs]


def out(verdict, *cites):
    return JudgeOutput(verdict=verdict, citations=[Citation(doc_id=d, sentence_ids=list(s)) for d, s in cites])


# -- the rule -------------------------------------------------------------------------------------


def test_strongest_sentence_wins_and_its_relation_is_the_larger_of_the_two_probabilities():
    scorer = FakeScorer({"Venules have a thin layer.": probs(entail=0.9), "Arterioles are thick.": probs(contra=0.6)})
    hit = NeiRecheck(scorer, tau=0.5).strongest(CLAIM, DOCS)
    assert (hit.doc_id, hit.index, hit.relation) == ("7", 1, Relation.ENTAILS) and hit.prob == pytest.approx(0.9)

    scorer = FakeScorer({"Capillaries are tiny.": probs(contra=0.8), "Venules have a thin layer.": probs(entail=0.3)})
    hit = NeiRecheck(scorer, tau=0.5).strongest(CLAIM, DOCS)
    assert (hit.index, hit.relation) == (2, Relation.CONTRADICTS)


def test_find_applies_a_strict_threshold_and_the_strongest_sentence_is_scored_once():
    scorer = FakeScorer({"Venules have a thin layer.": probs(entail=0.8)})
    rule = NeiRecheck(scorer, tau=0.8)
    assert rule.find(CLAIM, DOCS) is None  # 0.8 is not above 0.8
    assert NeiRecheck(scorer, tau=0.79).find(CLAIM, DOCS) is not None
    rule.strongest(CLAIM, DOCS)
    rule.strongest(CLAIM, DOCS)
    assert scorer.calls == 2  # one for `rule`, one for the second NeiRecheck; repeats were memoised


def test_no_sentences_means_no_hit():
    assert NeiRecheck(FakeScorer({}), tau=0.5).strongest(CLAIM, []) is None


# -- in the critic ----------------------------------------------------------------------------------


def test_a_nei_verdict_is_flagged_with_feedback_naming_the_sentence():
    rule = NeiRecheck(FakeScorer({"Venules have a thin layer.": probs(entail=0.93)}), tau=0.9)
    report = Critic(None, rule).check(CLAIM, DOCS, out(Verdict.NOT_ENOUGH_INFO))
    assert not report.passed and report.nei_hit.index == 1
    assert 'sentence 1 of document 7 ("Venules have a thin layer.") may support the claim' in report.feedback
    assert report.faithful is None  # a NOT_ENOUGH_INFO verdict has no faithfulness


def test_a_contradicting_sentence_is_described_as_such():
    rule = NeiRecheck(FakeScorer({"Arterioles are thick.": probs(contra=0.95)}), tau=0.9)
    report = Critic(None, rule).check(CLAIM, DOCS, out(Verdict.NOT_ENOUGH_INFO))
    assert "may contradict the claim" in report.feedback


def test_a_nei_verdict_passes_when_no_sentence_clears_the_threshold_or_there_is_no_rule():
    weak = NeiRecheck(FakeScorer({"Venules have a thin layer.": probs(entail=0.6)}), tau=0.9)
    assert Critic(None, weak).check(CLAIM, DOCS, out(Verdict.NOT_ENOUGH_INFO)).passed
    assert Critic(None, None).check(CLAIM, DOCS, out(Verdict.NOT_ENOUGH_INFO)).passed


def test_the_rule_never_touches_decisive_verdicts():
    rule = NeiRecheck(FakeScorer({"Venules have a thin layer.": probs(entail=0.99)}), tau=0.5)
    report = Critic(None, rule).check(CLAIM, DOCS, out(Verdict.SUPPORT, ("7", [1])))
    assert report.passed and report.nei_hit is None


# -- in the retry loop ----------------------------------------------------------------------------------


class FakeJudge:
    def __init__(self, *outputs):
        self.outputs, self.problems = list(outputs), []

    def judge(self, claim, docs, problem=None):
        self.problems.append(problem)
        return JudgeResult(self.outputs.pop(0), "ok", Usage(api_calls=1, model_calls=1))


class FakeChecker:
    name = "fake"

    def relation(self, claim, sentences):
        return RelationResult(Relation.ENTAILS, Usage(model_calls=1))


def test_a_flagged_nei_is_retried_with_the_sentence_in_the_feedback_and_can_be_corrected():
    rule = NeiRecheck(FakeScorer({"Venules have a thin layer.": probs(entail=0.95)}), tau=0.9)
    judge = FakeJudge(out(Verdict.NOT_ENOUGH_INFO), out(Verdict.SUPPORT, ("7", [1])))
    outcome = verify_claim(build_verify_graph(judge, Critic(FakeChecker(), rule)), CLAIM, DOCS)
    assert outcome.initial_verdict is Verdict.NOT_ENOUGH_INFO and outcome.final_keep is Verdict.SUPPORT
    assert outcome.flagged and outcome.final_passed and len(outcome.attempts) == 2
    assert "sentence 1 of document 7" in judge.problems[1]


def test_a_confirmed_nei_after_the_retry_is_kept_and_the_critic_still_objects():
    rule = NeiRecheck(FakeScorer({"Venules have a thin layer.": probs(entail=0.95)}), tau=0.9)
    judge = FakeJudge(out(Verdict.NOT_ENOUGH_INFO), out(Verdict.NOT_ENOUGH_INFO))
    outcome = verify_claim(build_verify_graph(judge, Critic(None, rule)), CLAIM, DOCS)
    assert outcome.final_keep is Verdict.NOT_ENOUGH_INFO and not outcome.final_passed  # keep policy: still NEI
    assert outcome.final_abstain is Verdict.NOT_ENOUGH_INFO


# -- choosing the threshold from stored outcomes ----------------------------------------------------------


def record(gold, initial, prob, retry_verdict):
    attempt0 = {"verdict": initial, "critic": {"nei_hit": None if prob is None else {"prob": prob}}}
    attempts = [attempt0] + ([{"verdict": retry_verdict, "critic": None}] if prob is not None else [])
    return {"gold": gold, "initial": initial, "attempts": attempts}


def test_threshold_simulation_counts_fixes_and_harms_per_threshold_from_stored_retries():
    nei = "NOT_ENOUGH_INFO"
    records = [
        record("SUPPORT", nei, 0.97, "SUPPORT"),     # false NEI, strong hit, retry fixes it
        record("CONTRADICT", nei, 0.91, "CONTRADICT"),  # false NEI, hit, fixed
        record("SUPPORT", nei, 0.62, "SUPPORT"),     # false NEI, weak hit, fixed
        record(nei, nei, 0.55, "SUPPORT"),            # correct NEI, weak hit, retry flips it: harm
        record(nei, nei, 0.93, nei),                  # correct NEI, strong hit, judge stays NEI: no harm
        record(nei, nei, None, None),                 # correct NEI, never flagged
        record("SUPPORT", "SUPPORT", None, None),     # not an NEI verdict: ignored
    ]
    result = simulate_nei_tau(records, grid=(0.5, 0.6, 0.9, 0.95))
    assert result["table"]["0.5"] == {"flagged": 5, "fixed": 3, "harmed": 1, "net": 2}
    assert result["table"]["0.6"] == {"flagged": 4, "fixed": 3, "harmed": 0, "net": 3}
    assert result["table"]["0.9"] == {"flagged": 3, "fixed": 2, "harmed": 0, "net": 2}
    assert result["table"]["0.95"] == {"flagged": 1, "fixed": 1, "harmed": 0, "net": 1}
    assert result["tau"] == 0.6  # best net with harms <= fixes
    assert (result["nei_verdicts"], result["false_nei"], result["correct_nei"]) == (6, 3, 3)


def test_the_rule_is_left_off_when_no_threshold_has_harms_at_most_fixes():
    nei = "NOT_ENOUGH_INFO"
    records = [record(nei, nei, 0.99, "SUPPORT"), record("SUPPORT", nei, 0.99, nei)]
    assert simulate_nei_tau(records, grid=(0.5, 0.9))["tau"] is None


def test_ties_resolve_to_the_stricter_threshold():
    records = [record("SUPPORT", "NOT_ENOUGH_INFO", 0.99, "SUPPORT")]
    assert simulate_nei_tau(records, grid=(0.5, 0.9, 0.95))["tau"] == 0.95


def test_nei_subset_stats_isolate_the_claims_the_rule_acts_on():
    gold = ["SUPPORT", "NOT_ENOUGH_INFO", "SUPPORT", "CONTRADICT"]
    initial = ["NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO", "SUPPORT", "CONTRADICT"]
    flagged = [True, True, False, False]
    final = ["SUPPORT", "SUPPORT", "SUPPORT", "CONTRADICT"]
    stats = _nei_subset_stats(gold, initial, flagged, final)
    assert stats["claims"] == 2 and stats["flagged"] == 2 and stats["fixed"] == 1 and stats["harmed"] == 1
    assert _nei_subset_stats(["SUPPORT"], ["SUPPORT"], [False], ["SUPPORT"]) is None
