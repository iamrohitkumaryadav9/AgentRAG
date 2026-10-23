"""Structured contracts for the claim-verification pipeline.

The judge returns a verdict plus the sentences it relied on; the critic's
blind polarity check returns a single relation. Both are validated with
Pydantic at the boundary, so a malformed generation fails loudly (and is
cached as a failure) instead of silently corrupting a metric.
"""
from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Verdict(str, Enum):
    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


class Relation(str, Enum):
    """How a set of evidence sentences relates to a claim (for the blind polarity check)."""

    ENTAILS = "entails"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


# The relation that a verdict claims to have found; NOT_ENOUGH_INFO claims there is none.
EXPECTED_RELATION = {
    Verdict.SUPPORT: Relation.ENTAILS,
    Verdict.CONTRADICT: Relation.CONTRADICTS,
    Verdict.NOT_ENOUGH_INFO: Relation.NEUTRAL,
}


class Citation(BaseModel):
    """Sentences (0-based indices) of one retrieved abstract that the verdict rests on."""

    doc_id: str
    sentence_ids: list[int] = Field(default_factory=list)

    @field_validator("doc_id", mode="before")
    @classmethod
    def _doc_id_as_text(cls, value):
        return str(value)


class JudgeOutput(BaseModel):
    verdict: Verdict
    citations: list[Citation] = Field(default_factory=list)
    rationale: str = Field(default="", description="One or two sentences explaining the verdict")


class RelationOutput(BaseModel):
    relation: Relation


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_model_json(model: type[BaseModel], raw: str):
    """Validate a model response, tolerating a markdown code fence around the JSON."""
    text = _FENCE_RE.sub("", raw.strip()).strip()
    return model.model_validate_json(text)
