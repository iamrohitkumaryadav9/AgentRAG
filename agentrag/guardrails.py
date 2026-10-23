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

# Function words carry no topical signal, so they'd inflate coverage for any
# question regardless of whether the corpus actually covers it.
_STOPWORDS = {
    "what", "which", "who", "whom", "whose", "why", "how", "when", "where",
    "the", "and", "for", "are", "was", "were", "been", "being", "have", "has",
    "had", "does", "did", "should", "would", "could", "can", "will", "shall",
    "this", "that", "these", "those", "with", "from", "into", "about", "than",
    "then", "there", "their", "them", "they", "you", "your", "its", "it's",
    "used", "use", "using", "make", "makes", "made", "get", "gets", "got",
    "system", "systems", "thing", "things", "way", "ways", "main", "commonly",
    "good", "bad", "big", "small", "new", "old", "work", "works", "need",
}


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


def query_coverage(query: str, context_chunks: List[Chunk]) -> float:
    """Fraction of the query's topical terms that appear in the retrieved context.

    This is the relevance gate that decides whether the system should attempt
    an answer at all. It is deliberately used *instead of* a raw similarity
    threshold: similarity scores are scale-dependent (they shift with chunk
    size, corpus size, and retrieval backend), so a fixed score floor tuned on
    one configuration silently breaks on another. Term coverage is normalised
    to [0,1] by construction and behaves consistently across backends.

    Empirically on this corpus, in-scope questions score >= 0.50 while
    out-of-scope questions score <= 0.25, leaving a wide margin around the
    default threshold.
    """
    terms = {w.lower() for w in _WORD_RE.findall(query)} - _STOPWORDS
    if not terms:
        return 1.0
    if not context_chunks:
        return 0.0

    context = " ".join(c.text.lower() for c in context_chunks)
    return round(sum(1 for t in terms if t in context) / len(terms), 3)


def check_citations(citations, context_chunks: List[Chunk]) -> list[str]:
    """Verify every cited source actually appears in the retrieved context.

    Structured output makes this check mechanical rather than heuristic: a
    model that fabricates a plausible-looking source name is caught here,
    which is one of the more insidious RAG failure modes because the answer
    still *looks* properly attributed.
    """
    valid_sources = {c.source for c in context_chunks}
    return [c.source for c in citations if c.source not in valid_sources]
