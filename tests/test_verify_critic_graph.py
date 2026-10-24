import pytest

from agentrag.verify.common import RelationResult, Usage
from agentrag.verify.critic import Critic, resolve_citations
from agentrag.verify.data import Doc
from agentrag.verify.graph import build_verify_graph, verify_claim
from agentrag.verify.judge import JudgeResult
from agentrag.verify.nli import NliPolarityChecker, decide_relation
from agentrag.verify.prompts import config_fingerprint, render_abstracts, render_judge, render_relation
from agentrag.verify.schemas import Citation, JudgeOutput, Relation, Verdict

DOCS = [
    Doc("7", "Vessel walls", ("Venules have a thin smooth layer.", "Arterioles have a thick layer.", "Unrelated.")),
    Doc("9", "Other", ("Something else entirely.",)),
]
CLAIM = "Venules have a thinner smooth layer than arterioles."


def out(verdict, *cites, rationale="r"):
    return JudgeOutput(verdict=verdict, citations=[Citation(doc_id=d, sentence_ids=list(s)) for d, s in cites],
                       rationale=rationale)


class FakeChecker:
    name = "fake"

    def __init__(self, *relations, status="ok"):
        self.relations, self.status, self.calls = list(relations), status, []

    def relation(self, claim, sentences):
        self.calls.append((claim, list(sentences)))
        return RelationResult(self.relations.pop(0), Usage(model_calls=1), status=self.status)


# -- critic ------------------------------------------------------------------------


def test_support_with_entailing_citations_passes_and_is_faithful():
    checker = FakeChecker(Relation.ENTAILS)
    report = Critic(checker).check(CLAIM, DOCS, out(Verdict.SUPPORT, ("7", [0, 1])))
    assert report.passed and report.polarity_ok and report.faithful is True
    assert report.cited_sentences == ["Venules have a thin smooth layer.", "Arterioles have a thick layer."]


def test_the_checker_is_blind_it_sees_only_the_claim_and_the_cited_sentences():
    checker = FakeChecker(Relation.ENTAILS)
    Critic(checker).check(CLAIM, DOCS, out(Verdict.SUPPORT, ("7", [0]), rationale="the secret rationale"))
    claim, sentences = checker.calls[0]
    assert claim == CLAIM and sentences == ["Venules have a thin smooth layer."]
    assert "SUPPORT" not in claim and "secret" not in " ".join(sentences)


def test_polarity_mismatch_fails_with_specific_feedback_and_is_unfaithful():
    report = Critic(FakeChecker(Relation.CONTRADICTS)).check(CLAIM, DOCS, out(Verdict.SUPPORT, ("7", [0])))
    assert not report.passed and report.polarity_ok is False and report.faithful is False
    assert "SUPPORT" in report.feedback and "contradict" in report.feedback


def test_a_neutral_relation_also_fails_a_decisive_verdict():
    report = Critic(FakeChecker(Relation.NEUTRAL)).check(CLAIM, DOCS, out(Verdict.CONTRADICT, ("7", [2])))
    assert not report.passed and "do not settle" in report.feedback


def test_contradict_with_contradicting_citations_passes():
    assert Critic(FakeChecker(Relation.CONTRADICTS)).check(CLAIM, DOCS, out(Verdict.CONTRADICT, ("7", [1]))).passed


@pytest.mark.parametrize(
    ("citation", "message"),
    [
        (("12345", [0]), "not among the abstracts"),
        (("7", [9]), "which has 3 sentences"),
        (("7", [-1]), "which has 3 sentences"),
    ],
)
def test_invalid_citations_fail_the_rules_without_calling_the_checker(citation, message):
    checker = FakeChecker()
    report = Critic(checker).check(CLAIM, DOCS, out(Verdict.SUPPORT, citation))
    assert not report.passed and message in report.feedback and checker.calls == []
    assert report.faithful is False


def test_a_decisive_verdict_with_no_citation_fails():
    report = Critic(FakeChecker()).check(CLAIM, DOCS, out(Verdict.SUPPORT))
    assert not report.passed and "without citing any sentence" in report.feedback


def test_not_enough_info_passes_without_a_polarity_check():
    checker = FakeChecker()
    report = Critic(checker).check(CLAIM, DOCS, out(Verdict.NOT_ENOUGH_INFO))
    assert report.passed and report.faithful is None and checker.calls == []


def test_a_rules_only_critic_accepts_valid_citations_and_never_checks_polarity():
    report = Critic(None).check(CLAIM, DOCS, out(Verdict.SUPPORT, ("7", [0])))
    assert report.passed and report.polarity_ok is None and report.faithful is None


def test_an_inconclusive_checker_does_not_penalise_the_verdict():
    checker = FakeChecker(Relation.NEUTRAL, status="blocked")
    report = Critic(checker).check(CLAIM, DOCS, out(Verdict.SUPPORT, ("7", [0])))
    assert report.passed and report.polarity_ok is None and report.checker_status == "blocked"


def test_resolve_citations_dedupes_and_keeps_order():
    valid, failures = resolve_citations(out(Verdict.SUPPORT, ("7", [1, 0]), ("7", [1]), ("9", [0])), DOCS)
    assert valid == [("7", 1), ("7", 0), ("9", 0)] and failures == []


# -- retry graph ---------------------------------------------------------------------


class FakeJudge:
    def __init__(self, *results):
        self.results, self.problems = list(results), []

    def judge(self, claim, docs, problem=None):
        self.problems.append(problem)
        return self.results.pop(0)


def ok(output, calls=1):
    return JudgeResult(output, "ok", Usage(api_calls=calls, model_calls=1, cost_usd=0.01))


def run(judge, critic, max_attempts=2):
    return verify_claim(build_verify_graph(judge, critic, max_attempts), CLAIM, DOCS)


def test_no_critic_means_one_judge_call_that_is_always_accepted():
    judge = FakeJudge(ok(out(Verdict.SUPPORT, ("7", [0]))))
    outcome = run(judge, None)
    assert len(outcome.attempts) == 1 and outcome.final_passed and not outcome.flagged
    assert outcome.final_keep is outcome.final_abstain is Verdict.SUPPORT


def test_a_passing_first_answer_is_not_retried():
    judge = FakeJudge(ok(out(Verdict.SUPPORT, ("7", [0]))))
    outcome = run(judge, Critic(FakeChecker(Relation.ENTAILS)))
    assert len(outcome.attempts) == 1 and not outcome.flagged and outcome.final_passed


def test_a_failed_check_retries_once_with_the_critics_feedback_and_can_fix_the_verdict():
    judge = FakeJudge(ok(out(Verdict.SUPPORT, ("7", [0]))), ok(out(Verdict.CONTRADICT, ("7", [1]))))
    outcome = run(judge, Critic(FakeChecker(Relation.CONTRADICTS, Relation.CONTRADICTS)))
    assert len(outcome.attempts) == 2 and outcome.flagged and outcome.final_passed
    assert outcome.initial_verdict is Verdict.SUPPORT and outcome.final_keep is Verdict.CONTRADICT
    assert judge.problems[0] is None and "contradict" in judge.problems[1]


def test_retries_are_bounded_and_the_two_failure_policies_differ():
    judge = FakeJudge(ok(out(Verdict.SUPPORT, ("7", [0]))), ok(out(Verdict.SUPPORT, ("7", [0]))))
    outcome = run(judge, Critic(FakeChecker(Relation.CONTRADICTS, Relation.CONTRADICTS)))
    assert len(outcome.attempts) == 2 and not outcome.final_passed
    assert outcome.final_keep is Verdict.SUPPORT and outcome.final_abstain is Verdict.NOT_ENOUGH_INFO


def test_max_attempts_one_disables_the_retry():
    judge = FakeJudge(ok(out(Verdict.SUPPORT, ("7", [0]))))
    outcome = run(judge, Critic(FakeChecker(Relation.CONTRADICTS)), max_attempts=1)
    assert len(outcome.attempts) == 1 and outcome.flagged and not outcome.final_passed


def test_an_unusable_judge_response_ends_the_run_as_not_enough_info_without_a_critic_call():
    checker = FakeChecker()
    judge = FakeJudge(JudgeResult(None, "invalid", Usage(api_calls=2, model_calls=2)))
    outcome = run(judge, Critic(checker))
    assert outcome.initial_verdict is Verdict.NOT_ENOUGH_INFO and not outcome.final_passed
    assert checker.calls == [] and len(outcome.attempts) == 1


def test_usage_adds_up_across_judge_and_critic_calls_and_attempts():
    judge = FakeJudge(ok(out(Verdict.SUPPORT, ("7", [0])), calls=1), ok(out(Verdict.CONTRADICT, ("7", [1])), calls=1))
    outcome = run(judge, Critic(FakeChecker(Relation.CONTRADICTS, Relation.CONTRADICTS)))
    assert outcome.usage.api_calls == 2 and outcome.usage.model_calls == 2 + 2  # 2 judge + 2 critic checks
    assert outcome.usage.cost_usd == pytest.approx(0.02)


# -- NLI checker and prompts ---------------------------------------------------------


@pytest.mark.parametrize(
    ("probs", "tau", "expected"),
    [
        ({"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}, 0.8, Relation.ENTAILS),
        ({"entailment": 0.7, "neutral": 0.2, "contradiction": 0.1}, 0.8, Relation.NEUTRAL),  # below tau
        ({"entailment": 0.05, "neutral": 0.05, "contradiction": 0.9}, 0.8, Relation.CONTRADICTS),
        ({"entailment": 0.2, "neutral": 0.7, "contradiction": 0.1}, 0.3, Relation.NEUTRAL),  # below tau
        ({"entailment": 0.4, "neutral": 0.5, "contradiction": 0.1}, 0.3, Relation.ENTAILS),  # neutral need not lose
        ({"entailment": 0.4, "neutral": 0.2, "contradiction": 0.4}, 0.3, Relation.ENTAILS),  # tie goes to entailment
    ],
)
def test_nli_decision_rule(probs, tau, expected):
    assert decide_relation(probs, tau) is expected


def test_nli_checker_concatenates_the_cited_sentences_as_premise_and_uses_the_claim_as_hypothesis():
    seen = []

    def scorer(premise, hypothesis):
        seen.append((premise, hypothesis))
        return {"entailment": 0.1, "neutral": 0.1, "contradiction": 0.8}

    result = NliPolarityChecker(scorer, tau_entail=0.5).relation(CLAIM, ["First.", "Second."])
    assert result.relation is Relation.CONTRADICTS and seen == [("First. Second.", CLAIM)]
    assert result.usage.model_calls == 1 and result.usage.api_calls == 0


def test_judge_prompt_numbers_sentences_and_adds_feedback_only_on_retry():
    prompt = render_judge(CLAIM, DOCS)
    assert "[Doc 7] Vessel walls" in prompt and "  [1] Arterioles have a thick layer." in prompt
    assert "reviewer" not in prompt
    assert "found a problem: bad citation" in render_judge(CLAIM, DOCS, "bad citation")
    assert render_abstracts(DOCS).count("[Doc ") == 2


def test_relation_prompt_contains_no_verdict_or_rationale_slot():
    prompt = render_relation(CLAIM, ["A sentence."])
    assert "A sentence." in prompt and CLAIM in prompt
    assert "verdict" not in prompt.lower() and "rationale" not in prompt.lower()


def test_config_fingerprint_is_stable_and_sensitive_to_settings():
    assert config_fingerprint(k=5, thinking="low") == config_fingerprint(k=5, thinking="low")
    assert config_fingerprint(k=5, thinking="low") != config_fingerprint(k=5, thinking="medium")


def test_the_two_directions_have_independent_thresholds():
    probs = {"entailment": 0.35, "neutral": 0.3, "contradiction": 0.35 - 0.001}
    assert decide_relation(probs, 0.3, 0.9) is Relation.ENTAILS
    assert decide_relation({"entailment": 0.1, "neutral": 0.5, "contradiction": 0.4}, 0.9, 0.3) is Relation.CONTRADICTS
    assert decide_relation({"entailment": 0.1, "neutral": 0.5, "contradiction": 0.4}, 0.3, 0.9) is Relation.NEUTRAL