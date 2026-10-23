"""The critic: rule checks on the citations plus a *blind* polarity check.

Blind means the checker sees only the claim and the sentences the judge cited,
never the verdict or the rationale, so it cannot simply agree with the judge.
It reports how those sentences relate to the claim (entails / contradicts /
neutral); the critic then compares that relation with the one the verdict
asserts. Two interchangeable checkers exist: an LLM call and a local NLI model.

Scope, stated plainly: the critic can catch a verdict whose own cited evidence
points the other way (the classic SUPPORT-vs-CONTRADICT inversion) or cites
nothing real. It cannot notice evidence the judge *failed* to cite, so a wrong
NOT_ENOUGH_INFO passes unchallenged.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .common import RelationResult, Usage
from .data import Doc
from .gemini import GeminiClient, InvalidModelOutput
from .prompts import RELATION_VERSION, render_relation
from .schemas import EXPECTED_RELATION, JudgeOutput, Relation, RelationOutput, Verdict

_PHRASE = {Relation.ENTAILS: "support", Relation.CONTRADICTS: "contradict", Relation.NEUTRAL: "do not settle"}


class PolarityChecker(Protocol):
    name: str

    def relation(self, claim: str, sentences: Sequence[str]) -> RelationResult: ...


class LlmPolarityChecker:
    name = "llm"

    def __init__(self, client: GeminiClient, prompt_version: str = RELATION_VERSION) -> None:
        self.client, self.prompt_version = client, prompt_version

    def relation(self, claim: str, sentences: Sequence[str]) -> RelationResult:
        prompt = render_relation(claim, sentences)
        try:
            result = self.client.generate(prompt, RelationOutput, prompt_version=self.prompt_version, label="relation")
        except InvalidModelOutput as err:
            usage = Usage.from_responses(err.responses, err.api_calls)
            return RelationResult(Relation.NEUTRAL, usage, status="invalid")
        usage = Usage.from_call(result)
        if result.parsed is None:
            return RelationResult(Relation.NEUTRAL, usage, status="blocked")
        return RelationResult(result.parsed.relation, usage)


@dataclass
class CriticReport:
    passed: bool
    rule_failures: list[str] = field(default_factory=list)
    cited_sentences: list[str] = field(default_factory=list)
    relation: Relation | None = None
    expected: Relation | None = None
    polarity_ok: bool | None = None  # None when no polarity check ran
    checker_status: str = "n/a"  # "ok" | "blocked" | "invalid" | "n/a"
    feedback: str | None = None
    usage: Usage = field(default_factory=Usage)

    @property
    def faithful(self) -> bool | None:
        """Do the cited sentences entail the verdict? None when undefined (NOT_ENOUGH_INFO, or no checker)."""
        if self.expected is None or self.expected is Relation.NEUTRAL:
            return None
        if self.rule_failures:
            return False  # a verdict resting on invalid or missing citations is not faithful
        return self.polarity_ok


def resolve_citations(output: JudgeOutput, docs: Sequence[Doc]) -> tuple[list[tuple[str, int]], list[str]]:
    """Valid (doc_id, sentence_index) pairs the judge cited, and a failure message for each invalid one."""
    by_id = {d.id: d for d in docs}
    valid: list[tuple[str, int]] = []
    failures: list[str] = []
    for citation in output.citations:
        doc = by_id.get(citation.doc_id)
        if doc is None:
            failures.append(f"you cited document {citation.doc_id}, which was not among the abstracts provided")
            continue
        for index in citation.sentence_ids:
            if 0 <= index < len(doc.sentences):
                if (doc.id, index) not in valid:
                    valid.append((doc.id, index))
            else:
                failures.append(
                    f"you cited sentence {index} of document {doc.id}, which has {len(doc.sentences)} sentences"
                )
    return valid, failures


class Critic:
    def __init__(self, checker: PolarityChecker | None = None) -> None:
        self.checker = checker

    def check(self, claim: str, docs: Sequence[Doc], output: JudgeOutput) -> CriticReport:
        by_id = {d.id: d for d in docs}
        valid, failures = resolve_citations(output, docs)
        sentences = [by_id[doc_id].sentences[i] for doc_id, i in valid]
        expected = EXPECTED_RELATION[output.verdict]

        if output.verdict is not Verdict.NOT_ENOUGH_INFO and not sentences and not failures:
            failures.append(f"you answered {output.verdict.value} without citing any sentence")
        report = CriticReport(passed=not failures, rule_failures=failures, cited_sentences=sentences, expected=expected)
        if failures:
            report.feedback = "; ".join(failures)
            return report
        if self.checker is None or output.verdict is Verdict.NOT_ENOUGH_INFO:
            return report

        result = self.checker.relation(claim, sentences)
        report.usage, report.relation, report.checker_status = result.usage, result.relation, result.status
        if result.status != "ok":
            return report  # inconclusive: the check could not run, so do not penalise the verdict
        report.polarity_ok = result.relation is expected
        if not report.polarity_ok:
            report.passed = False
            report.feedback = (
                f"your verdict was {output.verdict.value}, but read on their own the sentences you cited "
                f"{_PHRASE[result.relation]} the claim"
            )
        return report
