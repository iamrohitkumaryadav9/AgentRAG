"""Pluggable generation backends, returning validated structured output.

* ``GeminiLLM``      - real LLM calls via the Google Gemini API (used when
  ``GOOGLE_API_KEY`` is configured). Asks the model for JSON matching
  ``ANSWER_JSON_SCHEMA`` and validates the result with Pydantic.
* ``ExtractiveLLM``  - a deterministic, offline "generator" that composes an
  answer purely by selecting and stitching together the sentences most
  relevant to the query from the retrieved context. It never invents a fact
  that isn't present in the context, which makes it a useful zero-cost,
  zero-key fallback for local development, CI, and this repo's test suite.

Both return an ``AnswerPayload`` (answer + citations + confidence) rather
than a bare string, so the graph downstream of them is working with a
checked contract instead of prose it has to re-parse.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List

from .config import settings
from .ingest import Chunk
from .schemas import ANSWER_JSON_SCHEMA, AnswerPayload, Citation, parse_answer_payload

logger = logging.getLogger(__name__)


class LLM(ABC):
    backend_name: str = "base"

    @abstractmethod
    def generate(self, query: str, context_chunks: List[Chunk], feedback: str | None = None) -> AnswerPayload:
        ...


PROMPT_TEMPLATE = """You are a precise technical assistant. Answer the user's question using
ONLY the information in the context below. If the context does not contain
the answer, say you don't have enough information -- never invent facts.

Respond with a single JSON object matching this schema, and nothing else:
{schema}

- "answer": your answer, grounded strictly in the context.
- "citations": the [source#position] tags of the chunks you actually used.
- "confidence": a number in [0,1] reflecting how well the context supports your answer.

Context:
{context}

Question: {query}
{feedback_block}"""


class GeminiLLM(LLM):
    """Real LLM generation through the shared Gemini client.

    Requests go through ``agentrag.verify.gemini.GeminiClient``, so a transient 503 or a rate-limit 429
    is retried with exponential backoff (and a 429's own retry delay is honoured) instead of crashing the
    agent graph, and a malformed JSON reply gets one repair attempt. The request itself is unchanged: JSON
    mode with the schema described in the prompt, default thinking.
    """

    backend_name = "gemini"

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        client=None,
        *,
        max_retries: int = 5,
        timeout_seconds: float = 120.0,
    ) -> None:
        """``api_key``: when given, it is used as-is and the environment is never consulted (a hosted app passes
        each visitor's own key this way). Only when it is ``None`` does the CLI/API fall back to GOOGLE_API_KEY.
        ``max_retries`` / ``timeout_seconds`` bound how long a transient error can hold up a caller."""
        from .verify.gemini import GeminiClient, GenaiTransport

        if client is None:
            from .verify.cache import ResponseCache
            from .verify.ratelimit import Limits, RateLimiter

            if api_key is None:
                api_key = settings.google_api_key
            if not api_key:
                raise RuntimeError("no Gemini API key was provided")
            model = model_name or settings.gemini_model
            # No response cache (it would grow without bound in a service and must never be shared between users),
            # no spend ledger and no budget file: this path writes nothing to disk.
            client = GeminiClient(
                model,
                ResponseCache(None, enabled=False),
                RateLimiter(Limits(rpm=None, tpm=None, rpd=None)),  # rely on the server's 429 + backoff
                GenaiTransport(model, api_key, timeout_seconds=timeout_seconds, native_schema=False),
                thinking="default",
                max_retries=max_retries,
            )
        self._client = client

    def generate(self, query: str, context_chunks: List[Chunk], feedback: str | None = None) -> AnswerPayload:
        import json

        from .verify.gemini import InvalidModelOutput

        context = "\n\n".join(f"[{c.source}#{c.position}] {c.text}" for c in context_chunks)
        feedback_block = (
            f"\nA quality check rejected a previous answer for: {feedback}. Correct that in this attempt.\n"
            if feedback
            else ""
        )
        prompt = PROMPT_TEMPLATE.format(
            schema=json.dumps(ANSWER_JSON_SCHEMA, indent=2),
            context=context,
            query=query,
            feedback_block=feedback_block,
        )
        try:
            result = self._client.generate(prompt, AnswerPayload, prompt_version="qa-v1", label="qa")
        except InvalidModelOutput as err:
            # Still not valid JSON after a repair attempt: degrade to a plain-text answer, as before.
            return parse_answer_payload(err.raw)
        if result.parsed is None:  # the API blocked the response
            return AnswerPayload(
                answer="I don't have enough information to answer that (the model response was blocked).",
                citations=[],
                confidence=0.0,
            )
        return result.parsed


class ExtractiveLLM(LLM):
    """Offline fallback generator: extractive summarisation, no external calls."""

    backend_name = "extractive"

    def generate(self, query: str, context_chunks: List[Chunk], feedback: str | None = None) -> AnswerPayload:
        if not context_chunks:
            return AnswerPayload(
                answer="I don't have enough information in the knowledge base to answer that.",
                citations=[],
                confidence=0.0,
            )

        query_terms = {t.lower().strip("?.,") for t in query.split() if len(t) > 2}
        scored = []
        for chunk in context_chunks:
            for sentence in chunk.text.split(". "):
                sentence = sentence.strip().rstrip(".")
                if not sentence:
                    continue
                overlap = sum(1 for t in query_terms if t in sentence.lower())
                if overlap:
                    scored.append((overlap, sentence, chunk))

        if not scored:
            top = context_chunks[0]
            return AnswerPayload(
                answer=f"Based on {top.source}: {top.text[:280].rstrip()}...",
                citations=[Citation(source=top.source, position=top.position)],
                confidence=0.25,
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[:3]
        answer = ". ".join(sentence for _, sentence, _ in best) + "."

        seen: set[tuple[str, int]] = set()
        citations: List[Citation] = []
        for _, _, chunk in best:
            key = (chunk.source, chunk.position)
            if key not in seen:
                seen.add(key)
                citations.append(Citation(source=chunk.source, position=chunk.position))

        # Confidence scales with how strongly the selected sentences matched
        # the query -- a crude but honest self-assessment signal.
        max_overlap = max(score for score, _, _ in best)
        confidence = min(0.95, 0.4 + 0.15 * max_overlap)

        return AnswerPayload(answer=answer, citations=citations, confidence=round(confidence, 2))


def get_llm(backend: str = "auto") -> LLM:
    """Factory: GeminiLLM when a key is configured, else the offline ExtractiveLLM."""
    if backend == "extractive":
        return ExtractiveLLM()

    if backend in ("auto", "gemini"):
        try:
            return GeminiLLM()
        except Exception as exc:  # noqa: BLE001
            if backend == "gemini":
                raise
            logger.warning("Gemini backend unavailable (%s); falling back to offline extractive LLM.", exc)
            return ExtractiveLLM()

    raise ValueError(f"Unknown llm backend: {backend}")
