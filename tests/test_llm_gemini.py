"""The app's GeminiLLM goes through the shared client: retries, backoff and repair, tested with a fake transport."""
import json

import pytest

from agentrag.ingest import Chunk
from agentrag.llm import GeminiLLM, get_llm
from agentrag.schemas import AnswerPayload
from agentrag.verify.cache import ResponseCache
from agentrag.verify.gemini import GeminiClient, RateLimitError, RawResponse, TransientError
from agentrag.verify.ratelimit import Limits, RateLimiter

CHUNKS = [Chunk(id="1", source="rag.md", text="RAG combines retrieval with generation.", position=0)]
GOOD = json.dumps({"answer": "RAG combines retrieval and generation.",
                   "citations": [{"source": "rag.md", "position": 0}], "confidence": 0.9})


class FakeTransport:
    def __init__(self, *script):
        self.script, self.requests = list(script), []

    def call(self, prompt, schema, temperature, thinking):
        self.requests.append((prompt, schema.__name__, temperature, thinking))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _llm(*script, max_retries=5):
    transport = FakeTransport(*script)
    sleeps = []
    limiter = RateLimiter(Limits(rpm=None, tpm=None))
    client = GeminiClient("gemini-3.8-flash", ResponseCache(None, enabled=False), limiter, transport,
                          thinking="default", max_retries=max_retries, sleep=sleeps.append)
    llm = GeminiLLM(client=client)
    llm.transport, llm.sleeps = transport, sleeps
    return llm


def test_a_transient_503_is_retried_with_backoff_instead_of_crashing_the_graph():
    llm = _llm(TransientError("503 high demand"), TransientError("503 high demand"), RawResponse(GOOD))
    payload = llm.generate("What is RAG?", CHUNKS)
    assert isinstance(payload, AnswerPayload) and payload.confidence == 0.9
    assert payload.citations[0].source == "rag.md"
    assert len(llm.transport.requests) == 3 and len(llm.sleeps) == 2 and llm.sleeps[1] > 0


def test_a_rate_limit_429_is_retried_too():
    llm = _llm(RateLimitError("slow down", retry_after=0.0), RawResponse(GOOD))
    assert llm.generate("What is RAG?", CHUNKS).answer.startswith("RAG combines")


def test_giving_up_after_the_retry_budget_still_raises_so_the_failure_is_visible():
    llm = _llm(*[TransientError("503") for _ in range(3)], max_retries=2)
    with pytest.raises(TransientError):
        llm.generate("What is RAG?", CHUNKS)
    assert len(llm.transport.requests) == 3


def test_the_request_keeps_its_original_shape_and_includes_context_and_feedback():
    llm = _llm(RawResponse(GOOD))
    llm.generate("What is RAG?", CHUNKS, feedback="answer diverges from retrieved context")
    prompt, schema_name, temperature, thinking = llm.transport.requests[0]
    assert "[rag.md#0] RAG combines retrieval with generation." in prompt
    assert "Question: What is RAG?" in prompt and "answer diverges from retrieved context" in prompt
    assert '"answer"' in prompt  # the JSON schema is described in the prompt text
    assert (schema_name, thinking, temperature) == ("AnswerPayload", "default", 0.0)


def test_malformed_json_is_repaired_once_then_degrades_to_a_plain_text_answer():
    repaired = _llm(RawResponse("not json"), RawResponse(GOOD))
    assert repaired.generate("q", CHUNKS).confidence == 0.9

    hopeless = _llm(RawResponse("just a sentence"), RawResponse("still just a sentence"))
    payload = hopeless.generate("q", CHUNKS)
    assert payload.answer == "still just a sentence" and payload.confidence == 0.3  # the pre-existing degrade path


def test_a_blocked_response_becomes_an_honest_refusal_not_an_exception():
    payload = _llm(RawResponse("", blocked=True)).generate("q", CHUNKS)
    assert payload.confidence == 0.0 and "blocked" in payload.answer and payload.citations == []


def test_identical_questions_are_not_cached_across_calls_in_a_service():
    llm = _llm(RawResponse(GOOD), RawResponse(GOOD))
    llm.generate("same question", CHUNKS)
    llm.generate("same question", CHUNKS)
    assert len(llm.transport.requests) == 2  # memory must not grow with every distinct prompt


def test_auto_still_falls_back_to_the_offline_generator_without_a_key(monkeypatch):
    monkeypatch.setattr("agentrag.llm.settings", type("S", (), {"google_api_key": None, "gemini_model": "m"})())
    assert get_llm("auto").backend_name == "extractive"
    with pytest.raises(RuntimeError):
        get_llm("gemini")
