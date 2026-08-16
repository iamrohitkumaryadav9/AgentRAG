"""AgentRAG: a small, self-correcting multi-agent RAG assistant.

Modules:
    config       - runtime configuration via environment variables
    ingest       - document loading + chunking
    vectorstore  - pluggable retrieval backends (dense/FAISS, TF-IDF fallback)
    llm          - pluggable generation backends (Gemini, offline extractive)
    guardrails   - lightweight groundedness / safety checks on generated answers
    graph        - LangGraph multi-agent workflow (router -> retriever -> synthesizer -> critic)
    api          - FastAPI service exposing /ingest and /query
"""

__version__ = "0.1.0"
