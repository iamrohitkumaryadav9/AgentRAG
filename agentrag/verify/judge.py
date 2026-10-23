"""The judge: reads the retrieved abstracts and returns a cited verdict."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .common import Usage
from .data import Doc
from .gemini import GeminiClient, InvalidModelOutput
from .prompts import JUDGE_VERSION, render_judge
from .schemas import JudgeOutput


@dataclass
class JudgeResult:
    output: JudgeOutput | None
    status: str  # "ok" | "blocked" | "invalid"
    usage: Usage


class Judge:
    def __init__(self, client: GeminiClient, prompt_version: str = JUDGE_VERSION) -> None:
        self.client, self.prompt_version = client, prompt_version

    def judge(self, claim: str, docs: Sequence[Doc], problem: str | None = None) -> JudgeResult:
        prompt = render_judge(claim, docs, problem)
        try:
            result = self.client.generate(prompt, JudgeOutput, prompt_version=self.prompt_version, label="judge")
        except InvalidModelOutput as err:
            return JudgeResult(None, "invalid", Usage.from_responses(err.responses, err.api_calls))
        usage = Usage.from_call(result)
        if result.parsed is None:
            return JudgeResult(None, "blocked", usage)
        return JudgeResult(result.parsed, "ok", usage)
