"""Small types shared by the judge, the critic and the pipeline."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .cache import CachedResponse
from .gemini import CallResult, cost_usd
from .schemas import Relation


@dataclass
class Usage:
    """What a step cost. Token counts and latency come from the stored response, so a cached
    call reports the cost it *would* have had; ``api_calls`` counts only requests really sent."""

    api_calls: int = 0
    model_calls: int = 0  # logical model invocations (LLM or NLI), cached or not
    llm_calls: int = 0  # LLM responses involved (a repair attempt counts separately)
    prompt_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    thought_calls: int = 0  # LLM responses that used any thinking tokens
    max_call_usd: float = 0.0  # the most expensive single LLM response (a max, not a sum)
    max_thought_tokens: int = 0  # the most thinking used by a single LLM response (a max, not a sum)

    def __add__(self, other: Usage) -> Usage:
        merged = {
            name: getattr(self, name) + getattr(other, name)
            for name in vars(self)
            if name not in ("max_call_usd", "max_thought_tokens")
        }
        merged["max_call_usd"] = max(self.max_call_usd, other.max_call_usd)
        merged["max_thought_tokens"] = max(self.max_thought_tokens, other.max_thought_tokens)
        return Usage(**merged)

    @staticmethod
    def from_responses(responses: Sequence[CachedResponse], api_calls: int) -> Usage:
        usage = Usage(api_calls=api_calls, model_calls=len(responses), llm_calls=len(responses))
        for r in responses:
            call_cost = cost_usd(r.model, r.prompt_tokens, r.output_tokens, r.thought_tokens)
            usage.prompt_tokens += r.prompt_tokens
            usage.output_tokens += r.output_tokens
            usage.thought_tokens += r.thought_tokens
            usage.latency_ms += r.latency_ms
            usage.cost_usd += call_cost
            usage.thought_calls += 1 if r.thought_tokens > 0 else 0
            usage.max_call_usd = max(usage.max_call_usd, call_cost)
            usage.max_thought_tokens = max(usage.max_thought_tokens, r.thought_tokens)
        return usage

    @staticmethod
    def from_call(result: CallResult) -> Usage:
        return Usage.from_responses(result.responses or (result.response,), result.api_calls)


@dataclass
class RelationResult:
    """Outcome of a blind polarity check on some evidence sentences."""

    relation: Relation
    usage: Usage = field(default_factory=Usage)
    detail: dict = field(default_factory=dict)  # e.g. NLI class probabilities
    status: str = "ok"  # "ok" | "blocked" | "invalid"
