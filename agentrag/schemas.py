"""Structured output contracts.

The synthesizer agent does not return free text -- it returns an
``AnswerPayload`` validated by Pydantic. Forcing the model to fill a schema
(rather than parsing prose after the fact) is what makes downstream steps
reliable: the critic can check citations mechanically, the API can serialise
a stable response shape, and a malformed generation fails loudly at the
boundary instead of silently corrupting the answer.
"""
from __future__ import annotations

import json
import re
from typing import List

from pydantic import BaseModel, Field, field_validator


class Citation(BaseModel):
    """A pointer back to the chunk that supports a claim."""

    source: str = Field(..., description="Source document the claim came from")
    position: int = Field(0, ge=0, description="Chunk position within that document")


class AnswerPayload(BaseModel):
    """The structured contract every generation must satisfy."""

    answer: str = Field(..., description="The answer text, grounded in the provided context")
    citations: List[Citation] = Field(default_factory=list, description="Sources supporting the answer")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Self-reported confidence in [0,1]")

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("answer must not be empty")
        return value.strip()


# JSON Schema handed to the LLM so it knows exactly what shape to emit.
ANSWER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"source": {"type": "string"}, "position": {"type": "integer"}},
                "required": ["source"],
            },
        },
        "confidence": {"type": "number"},
    },
    "required": ["answer"],
}

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_answer_payload(raw: str) -> AnswerPayload:
    """Parse a model response into an AnswerPayload, tolerating minor drift.

    Models sometimes wrap JSON in prose or a ```json fence even when asked
    not to. Rather than failing the whole request on a cosmetic formatting
    slip, we extract the JSON object and validate that; if there is no
    parseable JSON at all, we fall back to treating the response as a plain
    answer string so the pipeline degrades instead of erroring out.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        return AnswerPayload.model_validate_json(text)
    except Exception:
        pass

    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            return AnswerPayload.model_validate(json.loads(match.group(0)))
        except Exception:
            pass

    return AnswerPayload(answer=text or "I don't have enough information to answer that.", confidence=0.3)
