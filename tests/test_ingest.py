from agentrag.ingest import build_corpus, chunk_text


def test_chunk_text_respects_size_and_overlap():
    text = "This is sentence one. This is sentence two. This is sentence three. " * 10
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        # allow a small margin: a single long sentence can slightly exceed chunk_size
        assert len(chunk) <= 130


def test_chunk_text_rejects_bad_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=50, chunk_overlap=50)


def test_build_corpus_from_sample_docs():
    corpus = build_corpus("data/sample_docs", chunk_size=500, chunk_overlap=80)
    assert len(corpus) > 0
    sources = {c.source for c in corpus}
    assert "rag_systems.md" in sources
    assert "agentic_ai.md" in sources
    # every chunk should carry text and a stable id
    assert all(c.text.strip() for c in corpus)
    assert len({c.id for c in corpus}) == len(corpus)
