"""Pluggable generation backends, returning validated structured output.

* ``GeminiLLM``      - real LLM calls via the Google Gemini API (used when
  ``GOOGLE_API_KEY`` is configured -- the same provider already used in the
  Quantis project's AI-insights feature). Asks the model for JSON matching
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
    """Real LLM generation via the current `google-genai` SDK."""

    backend_name = "gemini"

    def __init__(self, model_name: str | None = None, api_key: str | None = None) -> None:
        from google import genai

        api_key = api_key or settings.google_api_key
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name or settings.gemini_model

    def generate(self, query: str, context_chunks: List[Chunk], feedback: str | None = None) -> AnswerPayload:
        import json

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
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        return parse_answer_payload(response.text or "")


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
