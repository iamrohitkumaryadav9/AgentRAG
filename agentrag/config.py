"""Central configuration, read from environment variables (with sane defaults).

Keeping every tunable in one place makes the retrieval/generation trade-offs
(chunk size, top-k, backend choice) easy to reason about and to override
per-deployment without touching code -- a small nod to how these knobs would
be exposed in a real production AI service.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Chunking
    chunk_size: int = int(os.getenv("AGENTRAG_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("AGENTRAG_CHUNK_OVERLAP", "120"))

    # Retrieval
    top_k: int = int(os.getenv("AGENTRAG_TOP_K", "4"))
    vector_backend: str = os.getenv("AGENTRAG_VECTOR_BACKEND", "auto")  # auto | dense | tfidf

    # Generation
    llm_backend: str = os.getenv("AGENTRAG_LLM_BACKEND", "auto")  # auto | gemini | extractive
    gemini_model: str = os.getenv("AGENTRAG_GEMINI_MODEL", "gemini-1.5-flash")
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")

    # Agentic loop / guardrails
    max_synthesis_attempts: int = int(os.getenv("AGENTRAG_MAX_ATTEMPTS", "2"))
    grounding_threshold: float = float(os.getenv("AGENTRAG_GROUNDING_THRESHOLD", "0.18"))

    # Storage
    docs_dir: str = os.getenv("AGENTRAG_DOCS_DIR", "data/sample_docs")
    index_dir: str = os.getenv("AGENTRAG_INDEX_DIR", ".index")


settings = Settings()
