import logging
import os
from types import SimpleNamespace

import pytest

from agentrag import app_support as core
from agentrag.llm import ExtractiveLLM, GeminiLLM
from agentrag.schemas import AnswerPayload, Citation
from agentrag.verify.gemini import RateLimitError, TransientError

from .test_ingest_memory import make_pdf, sentences

SECRET = "AIzaSyFAKEKEYFORTESTING1234567890abcd"
MARKDOWN = (
    b"# Guide\n\nThe critic agent checks every answer against the retrieved passages. "
    b"A grounding score below the threshold sends the draft back to the synthesizer. "
    b"The router skips retrieval for small talk such as hello."
)


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("AGENTRAG_APP_MAX_CHUNKS_BM25", "AGENTRAG_APP_MAX_CHUNKS_HYBRID"):
        monkeypatch.delenv(name, raising=False)


# -- chunk caps -------------------------------------------------------------------------------------------


def test_default_caps_and_env_override(clean_env, monkeypatch):
    assert (core.chunk_cap("bm25"), core.chunk_cap("hybrid")) == (5000, 1000)
    monkeypatch.setenv("AGENTRAG_APP_MAX_CHUNKS_HYBRID", "40")
    assert core.chunk_cap("hybrid") == 40
    monkeypatch.setenv("AGENTRAG_APP_MAX_CHUNKS_HYBRID", "nonsense")
    assert core.chunk_cap("hybrid") == 1000


def test_build_index_stops_at_the_cap_and_says_so(clean_env, monkeypatch):
    monkeypatch.setenv("AGENTRAG_APP_MAX_CHUNKS_BM25", "12")
    files = [("big.txt", sentences(200, "big")), ("never.txt", sentences(5, "never"))]
    bundle = core.build_index(files, "bm25")
    assert len(bundle.chunks) == 12 and bundle.ingest.truncated
    summary = bundle.ingest.summary()
    assert "Indexed 12 chunks" in summary and "cap of 12 chunks was reached" in summary
    assert "big.txt: indexed 12 of" in summary and "never.txt" in summary


def test_build_index_reports_progress_ending_at_one(clean_env):
    seen = []
    core.build_index([("a.md", MARKDOWN)], "bm25", progress=lambda f, m: seen.append(f))
    assert seen == sorted(seen) and seen[-1] == 1.0 and max(seen) <= 1.0


def test_nothing_readable_raises_with_the_reasons(clean_env):
    with pytest.raises(core.NoIndexableText) as err:
        core.build_index([("empty.txt", b""), ("bad.pdf", b"not a pdf")], "bm25")
    assert "empty.txt" in str(err.value) and "bad.pdf" in str(err.value)


def test_pdf_md_and_txt_all_become_searchable(clean_env):
    files = [
        ("a.pdf", make_pdf("Zebrafish embryos are transparent.")),
        ("b.md", b"# Notes\n\nPlatypus eggs are leathery."),
        ("c.txt", b"Axolotls regenerate limbs."),
    ]
    bundle = core.build_index(files, "bm25")
    for query, source in [("zebrafish", "a.pdf"), ("platypus", "b.md"), ("axolotls", "c.txt")]:
        assert bundle.store.search(query, k=1)[0].chunk.source == source


def test_signature_changes_only_when_documents_or_backend_change():
    base = [("a.txt", b"one"), ("b.txt", b"two")]
    sig = core.documents_signature(base, "bm25")
    assert sig == core.documents_signature(list(base), "bm25")
    assert sig != core.documents_signature(base, "hybrid")
    assert sig != core.documents_signature([("a.txt", b"one"), ("b.txt", b"TWO")], "bm25")
    assert sig != core.documents_signature([("a.txt", b"one")], "bm25")


def test_dense_availability_matches_the_installed_packages():
    import importlib.util

    expected = all(importlib.util.find_spec(n) for n in ("sentence_transformers", "faiss"))
    assert core.dense_available() is bool(expected)


@pytest.mark.skipif(not core.dense_available(), reason="needs the dense extra")
def test_hybrid_index_uses_the_shared_model_and_its_own_cap(clean_env, monkeypatch):
    import numpy as np

    class TinyModel:  # hashed bag-of-words "embeddings": offline stand-in for bge-small
        calls = 0

        def encode(self, texts, convert_to_numpy=True):
            TinyModel.calls += 1
            out = np.zeros((len(texts), 64), dtype="float32")
            for i, text in enumerate(texts):
                for word in text.lower().split():
                    out[i, hash(word) % 64] += 1.0
            return out

    monkeypatch.setenv("AGENTRAG_APP_MAX_CHUNKS_HYBRID", "9")
    bundle = core.build_index([("big.txt", sentences(400, "big"))], "hybrid", dense_model=TinyModel())
    assert len(bundle.chunks) == 9 and bundle.ingest.truncated and bundle.store.backend_name == "hybrid-rrf"
    assert TinyModel.calls == 1
    assert bundle.store.search("sentence number 3", k=2)


# -- question handling ------------------------------------------------------------------------------------


def test_question_length_is_capped_and_empty_is_rejected():
    assert core.validate_question("  hi there  ") == "hi there"
    with pytest.raises(core.QuestionError, match="Please type"):
        core.validate_question("   ")
    with pytest.raises(core.QuestionError, match="1,000 characters"):
        core.validate_question("x" * 1001)
    assert len(core.validate_question("x" * 1000)) == 1000


class FakeGeminiClient:
    """Stands in for GeminiClient: records prompts, returns a canned grounded answer. No network."""

    def __init__(self, answer=None, error=None):
        self.prompts, self.answer, self.error = [], answer, error

    def generate(self, prompt, schema, prompt_version="", label=""):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return SimpleNamespace(parsed=self.answer)


def grounded_answer():
    return AnswerPayload(
        answer="The critic agent checks every answer against the retrieved passages.",
        citations=[Citation(source="guide.md", position=0)],
        confidence=0.9,
    )


def test_query_path_with_a_fake_gemini_client(clean_env):
    bundle = core.build_index([("guide.md", MARKDOWN)], "bm25")
    fake = FakeGeminiClient(answer=grounded_answer())
    report = core.answer_question(bundle, "What does the critic agent check?", GeminiLLM(client=fake))

    assert report.path == "answer" and report.generation == "gemini" and report.retrieval == "bm25"
    assert report.critic_passed is True and report.grounding_score >= 0.5 and report.attempts == 1
    assert [c.source for c in report.citations] == ["guide.md"] and "critic agent" in report.citations[0].snippet
    assert report.citations[0].found
    assert [s.agent for s in report.spans] == ["router", "retriever", "synthesizer", "critic"]
    assert all(s.ms >= 0 for s in report.spans) and report.total_ms > 0
    assert len(fake.prompts) == 1 and "critic agent" in fake.prompts[0]


def test_a_rejected_answer_is_retried_and_reported(clean_env):
    bundle = core.build_index([("guide.md", MARKDOWN)], "bm25")
    wrong = AnswerPayload(answer="Bananas orbit Jupiter quietly.", citations=[], confidence=0.9)
    fake = FakeGeminiClient(answer=wrong)
    report = core.answer_question(bundle, "What does the critic agent check?", GeminiLLM(client=fake))
    assert report.critic_passed is False and report.attempts > 1 and len(fake.prompts) == report.attempts
    assert report.critic_reason


def test_router_paths_direct_refuse_answer(clean_env):
    bundle = core.build_index([("guide.md", MARKDOWN)], "bm25")
    llm = ExtractiveLLM()
    assert core.answer_question(bundle, "hello", llm).path == "direct"
    assert core.answer_question(bundle, "What is the airspeed velocity of a swallow?", llm).path == "refuse"
    offline = core.answer_question(bundle, "What does the critic agent check?", llm)
    assert offline.path == "answer" and offline.generation == "extractive"


def test_a_fabricated_citation_is_flagged_not_hidden(clean_env):
    bundle = core.build_index([("guide.md", MARKDOWN)], "bm25")
    made_up = AnswerPayload(
        answer="The critic agent checks every answer against the retrieved passages.",
        citations=[Citation(source="ghost.md", position=3)],
        confidence=0.9,
    )
    llm = GeminiLLM(client=FakeGeminiClient(made_up))
    report = core.answer_question(bundle, "What does the critic agent check?", llm)
    assert report.bad_citations and not report.citations[0].found and report.critic_passed is False


# -- the API key ------------------------------------------------------------------------------------------


def test_no_key_means_offline_even_if_the_server_has_a_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "server-side-key-that-must-not-be-used")
    assert isinstance(core.make_llm(None), ExtractiveLLM)
    assert isinstance(core.make_llm("   "), ExtractiveLLM)


def test_the_key_goes_to_the_client_and_nowhere_else(monkeypatch, tmp_path, caplog):
    captured = {}

    class SpyTransport:
        def __init__(self, model, api_key, **kwargs):
            captured.update(model=model, api_key=api_key, **kwargs)

    monkeypatch.setattr("agentrag.verify.gemini.GenaiTransport", SpyTransport)
    monkeypatch.setenv("GOOGLE_API_KEY", "server-side-key-that-must-not-be-used")
    monkeypatch.chdir(tmp_path)
    env_before = dict(os.environ)

    with caplog.at_level(logging.DEBUG):
        llm = core.make_llm(f"  {SECRET}  ")

    assert isinstance(llm, GeminiLLM)
    assert captured["api_key"] == SECRET  # trimmed, and the server's own key was not substituted
    assert dict(os.environ) == env_before and SECRET not in os.environ.values()
    assert SECRET not in caplog.text
    assert list(tmp_path.iterdir()) == []  # no cache, ledger or budget file


def test_the_key_is_never_in_the_prompt_or_the_report(clean_env):
    bundle = core.build_index([("guide.md", MARKDOWN)], "bm25")
    fake = FakeGeminiClient(answer=grounded_answer())
    report = core.answer_question(bundle, "What does the critic agent check?", GeminiLLM(client=fake))
    assert SECRET not in repr(report) and not any(SECRET in p for p in fake.prompts)


def test_a_running_query_logs_nothing_secret_and_writes_no_files(clean_env, tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.DEBUG):
        bundle = core.build_index([("guide.md", MARKDOWN)], "bm25")
        llm = GeminiLLM(client=FakeGeminiClient(grounded_answer()))
        core.answer_question(bundle, "What does the critic agent check?", llm)
    assert SECRET not in caplog.text
    assert list(tmp_path.iterdir()) == []


# -- friendly errors --------------------------------------------------------------------------------------


class ApiError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (ApiError(400, "API key not valid. Please pass a valid API key."), "invalid_key"),
        (ApiError(403, "permission denied"), "invalid_key"),
        (RuntimeError("no Gemini API key was provided"), "invalid_key"),
        (RateLimitError("slow down"), "quota"),
        (ApiError(429, "RESOURCE_EXHAUSTED"), "quota"),
        (TransientError("503 UNAVAILABLE after 2 retries"), "unavailable"),
        (ApiError(503, "The model is overloaded"), "unavailable"),
        (TimeoutError("deadline"), "timeout"),
        (ValueError("something odd"), "unknown"),
    ],
)
def test_errors_map_to_actionable_messages(exc, kind):
    problem = core.friendly_error(exc, secret=SECRET)
    assert problem.kind == kind and problem.message


def test_error_text_never_echoes_the_key():
    exc = ApiError(400, f"API key not valid: {SECRET} (request key={SECRET})")
    problem = core.friendly_error(exc, secret=SECRET)
    assert SECRET not in problem.message and SECRET not in problem.detail and "[key hidden]" in problem.detail
    # even without being told the secret, key-shaped strings are removed
    assert SECRET not in core.friendly_error(exc).detail


def test_an_invalid_key_surfaces_through_the_whole_query_path(clean_env):
    bundle = core.build_index([("guide.md", MARKDOWN)], "bm25")
    llm = GeminiLLM(client=FakeGeminiClient(error=ApiError(400, f"API key not valid {SECRET}")))
    with pytest.raises(ApiError) as err:
        core.answer_question(bundle, "What does the critic agent check?", llm)
    assert core.friendly_error(err.value, secret=SECRET).kind == "invalid_key"
