"""The claim-verification loop as a LangGraph state graph: judge -> critic -> (accept | retry).

It reuses the pattern (and the ``traced_node`` telemetry) of the question-answering
graph in ``agentrag/graph.py``: draft, critique, and retry with specific feedback,
bounded by ``max_attempts``. The retrieval step is outside the graph because the
experiment retrieves once per claim and shares it across every arm.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, StateGraph

from ..observability import traced_node
from .common import Usage
from .critic import Critic, CriticReport
from .data import Doc
from .judge import Judge, JudgeResult
from .schemas import JudgeOutput, Verdict


@dataclass
class Attempt:
    judge_status: str
    output: JudgeOutput | None
    judge_usage: Usage
    report: CriticReport | None = None

    @property
    def verdict(self) -> Verdict:
        """The attempt's verdict; an unusable response (blocked / invalid) counts as NOT_ENOUGH_INFO."""
        return self.output.verdict if self.output else Verdict.NOT_ENOUGH_INFO

    @property
    def usage(self) -> Usage:
        return self.judge_usage + (self.report.usage if self.report else Usage())


class VerifyState(TypedDict, total=False):
    claim: str
    docs: list[Doc]
    attempts: list[Attempt]
    problem: str | None


@dataclass
class ClaimOutcome:
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def initial_verdict(self) -> Verdict:
        return self.attempts[0].verdict

    @property
    def flagged(self) -> bool:
        """Did the critic reject the first answer?"""
        first = self.attempts[0].report
        return first is not None and not first.passed

    @property
    def final_passed(self) -> bool:
        last = self.attempts[-1]
        return last.judge_status == "ok" and (last.report is None or last.report.passed)

    @property
    def final_keep(self) -> Verdict:
        """Policy 'keep': the last attempt's verdict, even if the critic still rejects it."""
        return self.attempts[-1].verdict

    @property
    def final_abstain(self) -> Verdict:
        """Policy 'abstain': if the critic still rejects the last attempt, fall back to NOT_ENOUGH_INFO."""
        if self.attempts[-1].report is not None and not self.attempts[-1].report.passed:
            return Verdict.NOT_ENOUGH_INFO
        return self.final_keep

    @property
    def usage(self) -> Usage:
        total = Usage()
        for attempt in self.attempts:
            total = total + attempt.usage
        return total


def build_verify_graph(judge: Judge, critic: Critic | None, max_attempts: int = 2):
    """Compile the loop. ``critic=None`` gives the no-critic baseline (one judge call, always accepted)."""

    def judge_node(state: VerifyState) -> VerifyState:
        attempts = list(state.get("attempts", []))
        with traced_node("verify_judge", attempt=len(attempts) + 1):
            result: JudgeResult = judge.judge(state["claim"], state["docs"], state.get("problem"))
        attempts.append(Attempt(result.status, result.output, result.usage))
        return {"attempts": attempts}

    def critic_node(state: VerifyState) -> VerifyState:
        attempts = list(state["attempts"])
        last = attempts[-1]
        if critic is not None and last.output is not None:
            with traced_node("verify_critic", attempt=len(attempts)):
                last.report = critic.check(state["claim"], state["docs"], last.output)
        return {"attempts": attempts, "problem": last.report.feedback if last.report else None}

    def route(state: VerifyState) -> str:
        last = state["attempts"][-1]
        if last.report is None or last.report.passed or len(state["attempts"]) >= max_attempts:
            return "end"
        return "retry"

    graph = StateGraph(VerifyState)
    graph.add_node("judge", judge_node)
    graph.add_node("critic", critic_node)
    graph.set_entry_point("judge")
    graph.add_edge("judge", "critic")
    graph.add_conditional_edges("critic", route, {"retry": "judge", "end": END})
    return graph.compile()


def verify_claim(compiled, claim: str, docs: Sequence[Doc]) -> ClaimOutcome:
    state = compiled.invoke({"claim": claim, "docs": list(docs), "attempts": [], "problem": None})
    return ClaimOutcome(attempts=state["attempts"])
