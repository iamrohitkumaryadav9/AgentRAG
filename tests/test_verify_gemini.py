import json

import pytest

from agentrag.verify.budget import Budget, BudgetExceeded, TripwireError
from agentrag.verify.cache import FIELDS, ResponseCache, find_secrets
from agentrag.verify.gemini import (
    GeminiClient,
    Halted,
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


def _client(tmp_path, transport, *, offline=False, limits=None, budget=None, **kwargs):
    sleeps = []
    cache = ResponseCache(tmp_path / "cache.jsonl", offline=offline)
    limiter = RateLimiter(limits or Limits(rpm=None, tpm=None), tmp_path / "ledger.json", budget=budget,
                          sleep=lambda s: None, day_fn=lambda: "2026-10-24")
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


def test_paid_equivalent_cost_counts_thinking_as_output():
    assert cost_usd("gemini-3.8-flash", 1_000_000, 0, 0) == pytest.approx(0.75)
    assert cost_usd("gemini-3.8-flash", 0, 500_000, 500_000) == pytest.approx(3.75)
    assert cost_usd("unknown-model", 10**6, 10**6, 10**6) == 0.0


def test_spend_is_charged_to_the_persistent_ledger_and_survives_a_new_client(tmp_path):
    budget = Budget(max_calls=100, max_inr=600.0)
    cheap = RawResponse(GOOD, prompt_tokens=2000, output_tokens=80)  # ~ $0.0018
    client = _client(tmp_path, FakeTransport(cheap, cheap), budget=budget)
    _gen(client, "one")
    _gen(client, "two")
    summary = client.limiter.spend_summary()
    assert summary["lifetime_calls"] == 2 and summary["lifetime_usd"] == pytest.approx(2 * 0.0018, rel=0.01)
    assert summary["lifetime_inr"] == pytest.approx(budget.inr(summary["lifetime_usd"]))
    reloaded = _client(tmp_path, FakeTransport(), budget=budget)
    assert reloaded.limiter.spend_summary()["lifetime_calls"] == 2


def test_the_hard_spend_cap_stops_before_the_request_that_would_cross_it(tmp_path):
    # a $0.005 cap leaves room for one call's worst-case reservation ($0.004) but, after a call that really cost
    # ~$0.0018, not for another reservation on top of it (0.0018 + 0.004 > 0.005)
    budget = Budget(max_calls=100, max_inr=0.005 * 600 / 6.8)
    cheap = RawResponse(GOOD, prompt_tokens=2000, output_tokens=80)
    transport = FakeTransport(cheap, cheap)
    client = _client(tmp_path, transport, budget=budget)
    _gen(client, "one")
    with pytest.raises(BudgetExceeded, match="spend cap"):
        _gen(client, "two")
    assert len(transport.requests) == 1


def test_the_hard_call_cap_counts_every_request_including_failed_attempts(tmp_path):
    budget = Budget(max_calls=2, max_inr=None)
    transport = FakeTransport(TransientError("503"), RawResponse(GOOD))
    client = _client(tmp_path, transport, budget=budget)
    _gen(client, "one")  # a failed attempt plus a successful one = 2 requests
    with pytest.raises(BudgetExceeded, match="call cap"):
        _gen(client, "two")
    assert len(transport.requests) == 2


def test_tripwire_on_an_expensive_call_halts_everything_but_the_response_is_cached(tmp_path):
    budget = Budget(max_calls=100, max_inr=600.0, tripwire_usd=0.004)
    expensive = RawResponse(GOOD, prompt_tokens=5000, output_tokens=100)  # ~ $0.0041
    transport = FakeTransport(expensive)
    client = _client(tmp_path, transport, budget=budget)
    with pytest.raises(TripwireError, match="cost"):
        _gen(client, "big prompt")
    assert "cost" in client.halt_reason
    assert client.cache.get(next(iter(client.cache._entries))) is not None  # paid for once, never again
    with pytest.raises(Halted):
        _gen(client, "another prompt")
    assert len(transport.requests) == 1


def test_tripwire_on_thinking_tokens(tmp_path):
    budget = Budget(max_calls=100, max_inr=600.0)
    client = _client(tmp_path, FakeTransport(RawResponse(GOOD, prompt_tokens=100, thought_tokens=1)), budget=budget)
    with pytest.raises(TripwireError, match="thinking"):
        _gen(client)


def test_normal_calls_pass_the_tripwire(tmp_path):
    client = _client(tmp_path, FakeTransport(RawResponse(GOOD, prompt_tokens=2230, output_tokens=84)),
                     budget=Budget())
    assert _gen(client).parsed.verdict is Verdict.SUPPORT


def test_concurrent_workers_share_one_client_safely(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    class Echo:
        def __init__(self):
            self.count = 0

        def call(self, prompt, schema, temperature, thinking):
            self.count += 1
            return RawResponse(GOOD, prompt_tokens=100, output_tokens=10)

    transport = Echo()
    client = _client(tmp_path, transport, budget=Budget(max_calls=1000))
    prompts = [f"claim {i}" for i in range(60)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda p: _gen(client, p), prompts + prompts))  # every prompt asked twice
    assert all(r.parsed.verdict is Verdict.SUPPORT for r in results)
    assert transport.count == 60  # each distinct prompt was paid for exactly once, even when asked concurrently
    assert client.limiter.spend_summary()["lifetime_calls"] == transport.count
    assert len(client.cache) == 60


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


def test_a_bookkeeping_failure_after_a_paid_call_never_discards_the_response(tmp_path, monkeypatch):
    client = _client(tmp_path, FakeTransport(RawResponse(GOOD, prompt_tokens=2000, output_tokens=80)),
                     budget=Budget())

    def boom(usd):
        raise OSError("ledger disk error")

    monkeypatch.setattr(client.limiter.ledger, "add_spend", boom)
    result = _gen(client, "paid for once")
    assert result.parsed.verdict is Verdict.SUPPORT
    assert _gen(client, "paid for once").from_cache  # it was cached despite the accounting failure
