import pytest
from pydantic import ValidationError

from agentrag.ingest import Chunk
from agentrag.guardrails import check_citations
from agentrag.llm import ExtractiveLLM
from agentrag.schemas import AnswerPayload, Citation, parse_answer_payload


def test_answer_payload_rejects_empty_answer():
    with pytest.raises(ValidationError):
        AnswerPayload(answer="   ")


def test_answer_payload_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        AnswerPayload(answer="ok", confidence=1.7)


def test_parse_clean_json():
    payload = parse_answer_payload('{"answer": "RAG combines retrieval and generation.", "confidence": 0.9}')
    assert payload.answer.startswith("RAG combines")
    assert payload.confidence == 0.9


def test_parse_json_wrapped_in_code_fence():
    raw = '```json\n{"answer": "Fenced answer", "confidence": 0.8}\n```'
    payload = parse_answer_payload(raw)
    assert payload.answer == "Fenced answer"


def test_parse_falls_back_to_plain_text():
    payload = parse_answer_payload("Just a plain sentence with no JSON at all.")
    assert "plain sentence" in payload.answer
    assert payload.confidence == 0.3


def test_extractive_llm_returns_valid_structured_payload():
    chunks = [Chunk(id="1", source="doc.md", text="RAG combines retrieval with generation.", position=0)]
    payload = ExtractiveLLM().generate("What is RAG?", chunks)

    assert isinstance(payload, AnswerPayload)
    assert payload.answer
    assert all(isinstance(c, Citation) for c in payload.citations)
    assert 0.0 <= payload.confidence <= 1.0


def test_check_citations_flags_fabricated_source():
    chunks = [Chunk(id="1", source="real.md", text="content", position=0)]
    citations = [Citation(source="real.md"), Citation(source="invented.md")]
    assert check_citations(citations, chunks) == ["invented.md"]
