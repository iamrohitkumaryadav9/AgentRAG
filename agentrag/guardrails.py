"""Lightweight quality & governance checks applied to every generated answer.

This is deliberately simple (regex/lexical heuristics, no extra model calls)
so it's fast, free, and fully deterministic -- but it implements the same
idea production AI-quality gates rely on:

1. Groundedness  - does the answer actually draw on the retrieved context,
   or does it look like an ungrounded/hallucinated response?
2. Refusal check - if nothing relevant was retrieved, the answer MUST say so
   instead of guessing.
3. PII leak check - a basic scan for emails/phone numbers being echoed back,
   as a stand-in for a real Responsible-AI/PII guardrail.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .ingest import Chunk

_WORD_RE = re.compile(r"[a-zA-Z]{3,}")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")

REFUSAL_PHRASES = ("don't have enough information", "not enough information", "cannot answer")


@dataclass
class GuardrailResult:
    passed: bool
    grounding_score: float
    reason: str


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def check_grounding(answer: str, context_chunks: List[Chunk], threshold: float = 0.18) -> GuardrailResult:
    """Reject answers whose content words barely overlap with the retrieved context.

    This is a cheap proxy for faithfulness/groundedness: it won't catch every
    subtle hallucination, but it reliably flags answers that ignore the
    supplied context entirely -- the most common and most damaging failure
    mode for a RAG system.
    """
    if any(phrase in answer.lower() for phrase in REFUSAL_PHRASES):
        # An honest refusal is always considered "passed" -- we'd rather the
        # system say "I don't know" than hallucinate.
        return GuardrailResult(passed=True, grounding_score=1.0, reason="honest refusal")

    if not context_chunks:
        return GuardrailResult(passed=False, grounding_score=0.0, reason="no supporting context retrieved")

    answer_words = _content_words(answer)
    context_words: set[str] = set()
    for chunk in context_chunks:
        context_words |= _content_words(chunk.text)

    if not answer_words:
        return GuardrailResult(passed=False, grounding_score=0.0, reason="empty or non-substantive answer")

    overlap = answer_words & context_words
    score = len(overlap) / len(answer_words)
    passed = score >= threshold
    reason = "sufficient overlap with retrieved context" if passed else "answer diverges from retrieved context"
    return GuardrailResult(passed=passed, grounding_score=round(score, 3), reason=reason)


def check_pii_leak(answer: str) -> list[str]:
    """Return a list of PII patterns detected in the answer (should normally be empty)."""
    findings = []
    if _EMAIL_RE.search(answer):
        findings.append("email_address")
    if _PHONE_RE.search(answer):
        findings.append("phone_number")
    return findings
