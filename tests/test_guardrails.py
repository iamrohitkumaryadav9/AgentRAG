from agentrag.guardrails import check_grounding, check_pii_leak
from agentrag.ingest import Chunk


def _chunk(text: str) -> Chunk:
    return Chunk(id="1", source="doc.md", text=text, position=0)


def test_grounded_answer_passes():
    context = [_chunk("RAG combines retrieval with generation using a vector store.")]
    answer = "RAG combines retrieval with generation using a vector store for grounding."
    result = check_grounding(answer, context, threshold=0.18)
    assert result.passed
    assert result.grounding_score > 0


def test_ungrounded_answer_fails():
    context = [_chunk("RAG combines retrieval with generation using a vector store.")]
    answer = "The stock market closed higher today amid strong earnings reports."
    result = check_grounding(answer, context, threshold=0.18)
    assert not result.passed


def test_honest_refusal_always_passes():
    result = check_grounding("I don't have enough information to answer that.", [], threshold=0.5)
    assert result.passed
    assert result.grounding_score == 1.0


def test_no_context_fails_if_not_a_refusal():
    result = check_grounding("The answer is definitely 42.", [], threshold=0.18)
    assert not result.passed


def test_pii_leak_detection():
    assert check_pii_leak("Contact me at rohit@example.com") == ["email_address"]
    assert check_pii_leak("Call 9876543210 for details") == ["phone_number"]
    assert check_pii_leak("No sensitive data here.") == []
