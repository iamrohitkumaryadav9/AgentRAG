"""Drives app.py headlessly with Streamlit's AppTest. Offline: a fake Gemini client stands in for the API."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from agentrag import app_support as core  # noqa: E402
from agentrag.llm import GeminiLLM  # noqa: E402
from agentrag.schemas import AnswerPayload, Citation  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")
SECRET = "AIzaSyFAKEKEYFORTESTING1234567890abcd"


def fresh() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


def texts(elements) -> str:
    return " ".join(str(e.value) for e in elements)


def use_samples(at: AppTest) -> AppTest:
    next(b for b in at.button if b.label == "Use sample docs").click()
    return at.run()


def ask(at: AppTest, question: str) -> AppTest:
    at.text_area[0].set_value(question)
    next(b for b in at.button if b.label == "Ask").click()
    return at.run()


def test_first_load_shows_offline_mode_and_prompts_for_documents():
    at = fresh()
    assert not at.exception
    assert "Offline mode" in texts(at.sidebar.info)
    assert "never stored" in texts(at.sidebar.caption)
    assert "Use sample docs" in [b.label for b in at.button]
    assert "Upload one or more documents" in texts(at.info)


def test_the_key_box_is_a_password_field():
    at = fresh()
    assert at.sidebar.text_input[0].proto.type == 1  # PASSWORD


def test_sample_docs_are_indexed_once_not_on_every_question(monkeypatch):
    builds = []
    real = core.build_index
    monkeypatch.setattr(core, "build_index", lambda *a, **k: builds.append(1) or real(*a, **k))
    at = use_samples(fresh())
    assert "Indexed" in texts(at.success) and len(builds) == 1
    at = ask(at, "What does the critic agent check?")
    at = ask(at, "How does the router decide?")
    assert len(builds) == 1 and not at.exception


def test_offline_answer_shows_everything_the_user_asked_for():
    at = use_samples(fresh())
    at = ask(at, "What does the critic agent check?")
    assert not at.exception and not at.error
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Router path"] == "answer" and metrics["Attempts"] == "1"
    assert metrics["Critic verdict"] in {"passed", "rejected"} and metrics["Grounding score"] != "—"
    page = texts(at.markdown) + texts(at.caption) + texts(at.code)
    assert "extractive" in page and "router ->" in page and "critic ->" in page
    assert any("chunk" in e.label for e in at.expander)  # citations with snippets


def test_refusal_and_empty_question_paths():
    at = use_samples(fresh())
    at = ask(at, "What is the airspeed velocity of a swallow?")
    assert {m.label: m.value for m in at.metric}["Router path"] == "refuse"
    at = ask(at, "   ")
    assert "Please type" in texts(at.warning)


def test_the_key_drives_gemini_mode_through_a_fake_client_and_stays_out_of_the_environment(monkeypatch):
    seen = {}

    class Fake:
        def generate(self, prompt, schema, prompt_version="", label=""):
            return SimpleNamespace(parsed=AnswerPayload(
                answer="The critic agent checks every answer against the retrieved passages.",
                citations=[Citation(source="architecture.md", position=0)], confidence=0.9))

    def fake_make_llm(api_key):
        seen["key"] = api_key
        return GeminiLLM(client=Fake())

    monkeypatch.setattr(core, "make_llm", fake_make_llm)
    env_before = dict(os.environ)
    at = use_samples(fresh())
    at.sidebar.text_input[0].set_value(SECRET).run()
    assert "Gemini mode" in texts(at.sidebar.success)
    at = ask(at, "What does the critic agent check?")
    assert seen["key"] == SECRET and not at.exception
    assert dict(os.environ) == env_before
    assert SECRET not in texts(at.markdown) + texts(at.caption) + texts(at.code) + texts(at.table)


def test_api_failures_become_friendly_errors_without_the_key(monkeypatch):
    class Boom:
        def generate(self, *a, **k):
            err = RuntimeError(f"API key not valid: {SECRET}")
            err.code = 400
            raise err

    monkeypatch.setattr(core, "make_llm", lambda key: GeminiLLM(client=Boom()))
    at = use_samples(fresh())
    at.sidebar.text_input[0].set_value(SECRET).run()
    at = ask(at, "What does the critic agent check?")
    assert not at.exception and "rejected this API key" in texts(at.error)
    assert SECRET not in texts(at.error) + texts(at.code)


def test_hybrid_toggle_is_disabled_with_a_tooltip_when_dense_is_missing(monkeypatch):
    monkeypatch.setattr(core, "dense_available", lambda: False)
    at = fresh()
    toggle = at.sidebar.toggle[0]
    assert toggle.disabled and toggle.label == "Hybrid (BM25 + bge-small)" and toggle.value is False
