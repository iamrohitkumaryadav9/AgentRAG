import json

import pytest

from agentrag.verify.cache import FIELDS, ResponseCache, find_secrets
from agentrag.verify.gemini import (
    BudgetExceeded,
    GeminiClient,
    InvalidModelOutput,
    RateLimitError,
    RawResponse,
    TransientError,
    cost_usd,
    translate_error,
)
from agentrag.verify.ratelimit import DailyBudgetExhausted, Limits, RateLimiter
from agentrag.verify.schemas import JudgeOutput, Verdict

GOOD = json.dumps({"verdict": "SUPPORT", "citations": [{"doc_id": 7, "sentence_ids": [2]}], "rationale": "ok"})


class FakeTransport:
    """Plays back a script of RawResponse objects / exceptions and counts requests."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests = []

    def call(self, prompt, schema, temperature, thinking):
        self.requests.append((prompt, temperature, thinking))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _client(tmp_path, transport, *, offline=False, limits=None, **kwargs):
    sleeps = []
    cache = ResponseCache(tmp_path / "cache.jsonl", offline=offline)
    limiter = RateLimiter(limits or Limits(rpm=None, tpm=None), tmp_path / "ledger.json", sleep=lambda s: None,
                          day_fn=lambda: "2026-10-24")
    client = GeminiClient("gemini-3.8-flash", cache, limiter, transport, sleep=sleeps.append, **kwargs)
    client.sleeps = sleeps
    return client


def _gen(client, prompt="claim?"):
    return client.generate(prompt, JudgeOutput, prompt_version="judge-v1", label="judge")


def test_a_miss_calls_the_api_once_then_the_same_request_is_served_from_cache(tmp_path):
    transport = FakeTransport(RawResponse(GOOD, prompt_tokens=100, output_tokens=20, thought_tokens=30))
    client = _client(tmp_path, transport)
    first = _gen(client)
    assert first.parsed.verdict is Verdict.SUPPORT and first.parsed.citations[0].doc_id == "7"
    assert not first.from_cache and first.api_calls == 1

    second = _gen(client)
    assert second.from_cache and second.api_calls == 0 and len(transport.requests) == 1
    assert second.response.thought_tokens == 30

    fresh = _client(tmp_path, FakeTransport())  # new process, same cache file, transport must stay unused
    assert _gen(fresh).from_cache


def test_requests_carry_the_configured_temperature_and_thinking_level(tmp_path):
    transport = FakeTransport(RawResponse(GOOD))
    _gen(_client(tmp_path, transport, thinking="medium", temperature=0.0))
    assert transport.requests[0][1:] == (0.0, "medium")


def test_changing_thinking_level_is_a_different_cache_entry(tmp_path):
    low = _client(tmp_path, FakeTransport(RawResponse(GOOD)), thinking="low")
    _gen(low)
    medium_transport = FakeTransport(RawResponse(GOOD))
    _gen(_client(tmp_path, medium_transport, thinking="medium"))
    assert len(medium_transport.requests) == 1  # not answered from the "low" entry


def test_invalid_json_gets_exactly_one_repair_attempt(tmp_path):
    transport = FakeTransport(RawResponse("not json at all"), RawResponse(GOOD))
    result = _gen(_client(tmp_path, transport))
    assert result.parsed.verdict is Verdict.SUPPORT and result.api_calls == 2
    assert "not valid JSON" in transport.requests[1][0]


def test_still_invalid_after_repair_raises_and_is_cached_so_it_is_not_retried(tmp_path):
    transport = FakeTransport(RawResponse("garbage"), RawResponse("still garbage"))
    client = _client(tmp_path, transport)
    with pytest.raises(InvalidModelOutput) as err:
        _gen(client)
    assert err.value.raw == "still garbage"
    with pytest.raises(InvalidModelOutput):
        _gen(client)
    assert len(transport.requests) == 2  # the rerun used the cache for both attempts


def test_markdown_fenced_json_is_accepted(tmp_path):
    result = _gen(_client(tmp_path, FakeTransport(RawResponse(f"```json\n{GOOD}\n```"))))
    assert result.parsed.verdict is Verdict.SUPPORT and result.api_calls == 1


def test_a_safety_block_is_a_cached_result_not_an_exception(tmp_path):
    transport = FakeTransport(RawResponse("", blocked=True, finish_reason="SAFETY"))
    client = _client(tmp_path, transport)
    result = _gen(client)
    assert result.blocked and result.parsed is None
    assert _gen(client).from_cache and len(transport.requests) == 1


def test_transient_errors_back_off_and_then_succeed(tmp_path):
    transport = FakeTransport(TransientError("503"), TransientError("503"), RawResponse(GOOD))
    client = _client(tmp_path, transport, base_delay=2.0)
    assert _gen(client).api_calls == 3
    assert len(client.sleeps) == 2 and client.sleeps[1] > client.sleeps[0] * 0.9  # delays grow (with jitter)


def test_transient_errors_give_up_after_max_retries(tmp_path):
    transport = FakeTransport(*[TransientError("503") for _ in range(3)])
    with pytest.raises(TransientError):
        _gen(_client(tmp_path, transport, max_retries=2))
    assert len(transport.requests) == 3


def test_a_rate_limit_429_cools_the_limiter_down_and_retries(tmp_path):
    class T:
        now, slept = 0.0, 0.0

    clock = T()

    def sleep(s):
        clock.now += s
        clock.slept += s

    transport = FakeTransport(RateLimitError("slow down", retry_after=25.0), RawResponse(GOOD))
    cache = ResponseCache(tmp_path / "c.jsonl")
    limiter = RateLimiter(Limits(rpm=None, tpm=None), clock=lambda: clock.now, sleep=sleep, day_fn=lambda: "d")
    client = GeminiClient("gemini-3.8-flash", cache, limiter, transport, sleep=sleep)
    assert _gen(client).api_calls == 2
    assert clock.slept >= 25


def test_a_daily_quota_429_stops_the_run_and_blocks_further_calls(tmp_path):
    transport = FakeTransport(RateLimitError("per day", daily=True))
    client = _client(tmp_path, transport)
    with pytest.raises(DailyBudgetExhausted):
        _gen(client)
    with pytest.raises(DailyBudgetExhausted):  # refused locally, no second request sent
        _gen(client, prompt="another claim?")
    assert len(transport.requests) == 1


def test_paid_equivalent_cost_counts_thinking_as_output_and_a_cap_can_stop_the_run(tmp_path):
    assert cost_usd("gemini-3.8-flash", 1_000_000, 0, 0) == pytest.approx(0.75)
    assert cost_usd("gemini-3.8-flash", 0, 500_000, 500_000) == pytest.approx(3.75)
    assert cost_usd("unknown-model", 10**6, 10**6, 10**6) == 0.0

    transport = FakeTransport(RawResponse(GOOD, prompt_tokens=1_000_000))
    client = _client(tmp_path, transport, max_usd=0.50)
    _gen(client)
    assert client.spent_usd == pytest.approx(0.75)
    with pytest.raises(BudgetExceeded):
        _gen(client, prompt="a different request")


def test_the_cache_file_contains_only_declared_fields_and_no_secrets(tmp_path):
    client = _client(tmp_path, FakeTransport(RawResponse(GOOD, model_version="gemini-3.8-flash")))
    _gen(client)
    text = (tmp_path / "cache.jsonl").read_text(encoding="utf-8")
    assert find_secrets(text) == []
    assert set(json.loads(text.splitlines()[0])) == set(FIELDS)


def test_unknown_thinking_level_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _client(tmp_path, FakeTransport(), thinking="off")


def test_error_translation_recognises_rate_limits_daily_quotas_and_transient_failures():
    body = {"error": {"details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"},
    ]}}
    err = translate_error(429, "RESOURCE_EXHAUSTED", "quota", body)
    assert isinstance(err, RateLimitError) and err.daily and err.retry_after == 34.0

    per_minute = {"error": {"details": [{"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}]}]}}
    err = translate_error(429, None, "rpm", per_minute)
    assert isinstance(err, RateLimitError) and not err.daily and err.retry_after is None

    assert isinstance(translate_error(503, "UNAVAILABLE", "busy", {}), TransientError)
    assert translate_error(400, "INVALID_ARGUMENT", "bad request", {}) is None
