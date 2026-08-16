"""Pluggable generation backends.

* ``GeminiLLM``      - real LLM calls via the Google Gemini API (used when
  ``GOOGLE_API_KEY`` is configured -- the same provider already used in the
  Quantis project's AI-insights feature).
* ``ExtractiveLLM``  - a deterministic, offline "generator" that composes an
  answer purely by selecting and stitching together the sentences most
  relevant to the query from the retrieved context. It never invents a fact
  that isn't present in the context, which makes it a useful zero-cost,
  zero-key fallback for local development, CI, and this repo's test suite.

Both implement the same ``generate(query, context)`` interface so the graph
in ``graph.py`` doesn't need to know which one is active.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List

from .config import settings
from .ingest import Chunk

logger = logging.getLogger(__name__)


class LLM(ABC):
    backend_name: str = "base"

    @abstractmethod
    def generate(self, query: str, context_chunks: List[Chunk], feedback: str | None = None) -> str:
        ...


PROMPT_TEMPLATE = """You are a precise technical assistant. Answer the user's question using
ONLY the information in the context below. If the context does not contain
the answer, say you don't have enough information -- never invent facts.

Context:
{context}

Question: {query}
{feedback_block}
Answer:"""


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

    def generate(self, query: str, context_chunks: List[Chunk], feedback: str | None = None) -> str:
        context = "\n\n".join(f"[{c.source}#{c.position}] {c.text}" for c in context_chunks)
        feedback_block = f"\n(Note: a quality check rejected a previous answer for: {feedback}. Fix that.)\n" if feedback else ""
        prompt = PROMPT_TEMPLATE.format(context=context, query=query, feedback_block=feedback_block)
        response = self._client.models.generate_content(model=self._model_name, contents=prompt)
        return (response.text or "").strip()


class ExtractiveLLM(LLM):
    """Offline fallback generator: extractive summarisation, no external calls."""

    backend_name = "extractive"

    def generate(self, query: str, context_chunks: List[Chunk], feedback: str | None = None) -> str:
        if not context_chunks:
            return "I don't have enough information in the knowledge base to answer that."

        query_terms = {t.lower() for t in query.split() if len(t) > 2}
        scored_sentences = []
        for chunk in context_chunks:
            for sentence in chunk.text.split(". "):
                sentence = sentence.strip().rstrip(".")
                if not sentence:
                    continue
                overlap = sum(1 for t in query_terms if t in sentence.lower())
                if overlap:
                    scored_sentences.append((overlap, sentence, chunk.source))

        if not scored_sentences:
            top = context_chunks[0]
            return f"Based on {top.source}: {top.text[:280].rstrip()}..."

        scored_sentences.sort(key=lambda item: item[0], reverse=True)
        best = scored_sentences[:3]
        answer = ". ".join(sentence for _, sentence, _ in best)
        sources = ", ".join(sorted({src for _, _, src in best}))
        return f"{answer}. (Source: {sources})"


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
