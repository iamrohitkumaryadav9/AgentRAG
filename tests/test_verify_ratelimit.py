import pytest

from agentrag.verify.ratelimit import DailyBudgetExhausted, Limits, RateLimiter, limits_from_env


class FakeTime:
    def __init__(self):
        self.now = 1000.0
        self.slept = 0.0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.slept += seconds
        self.now += seconds


def _limiter(limits, tmp_path=None, day="2026-10-24", ft=None):
    ft = ft or FakeTime()
    ledger = tmp_path / "ledger.json" if tmp_path else None
    limiter = RateLimiter(limits, ledger, clock=ft.clock, sleep=ft.sleep, day_fn=lambda: day)
    return limiter, ft


def test_requests_within_rpm_do_not_wait_and_the_next_one_waits_for_the_window():
    limiter, ft = _limiter(Limits(rpm=2, tpm=None))
    limiter.acquire()
    limiter.acquire()
    assert ft.slept == 0
    limiter.acquire()
    assert 60 <= ft.slept < 61  # waited for the first request to leave the 60 s window


def test_tokens_per_minute_are_enforced_with_real_usage_replacing_the_estimate():
    limiter, ft = _limiter(Limits(rpm=None, tpm=1000))
    limiter.acquire(600)
    limiter.record_tokens(100)  # the call turned out cheaper than estimated
    limiter.acquire(600)
    assert ft.slept == 0
    limiter.acquire(600)  # 100 + 600 + 600 > 1000 until the oldest entry expires
    assert ft.slept > 0


def test_a_request_larger_than_the_tpm_limit_is_rejected_up_front():
    limiter, _ = _limiter(Limits(rpm=None, tpm=500))
    with pytest.raises(ValueError):
        limiter.acquire(501)


def test_daily_budget_stops_cleanly_and_survives_a_restart(tmp_path):
    limiter, _ = _limiter(Limits(rpm=None, tpm=None, rpd=2), tmp_path)
    limiter.acquire()
    limiter.acquire()
    with pytest.raises(DailyBudgetExhausted):
        limiter.acquire()

    restarted, _ = _limiter(Limits(rpm=None, tpm=None, rpd=2), tmp_path)
    assert restarted.ledger.count_today() == 2
    with pytest.raises(DailyBudgetExhausted):
        restarted.acquire()

    next_day, _ = _limiter(Limits(rpm=None, tpm=None, rpd=2), tmp_path, day="2026-10-25")
    next_day.acquire()  # a new Pacific day resets the budget
    assert next_day.ledger.count_today() == 1


def test_hard_call_cap_applies_per_process_regardless_of_the_ledger():
    limiter, _ = _limiter(Limits(rpm=None, tpm=None, max_calls=1))
    limiter.acquire()
    with pytest.raises(DailyBudgetExhausted, match="hard call cap"):
        limiter.acquire()


def test_penalize_imposes_a_cooldown_from_a_server_429():
    limiter, ft = _limiter(Limits(rpm=None, tpm=None))
    limiter.penalize(30)
    limiter.acquire()
    assert 30 <= ft.slept < 31


def test_exhaust_today_blocks_further_requests_and_records_why(tmp_path):
    limiter, _ = _limiter(Limits(rpm=None, tpm=None), tmp_path)
    limiter.exhaust_today("quota GenerateRequestsPerDay exceeded")
    with pytest.raises(DailyBudgetExhausted, match="server reported"):
        limiter.acquire()
    again, _ = _limiter(Limits(rpm=None, tpm=None), tmp_path)
    assert again.ledger.exhausted_today() and "PerDay" in again.ledger._data["last_429"]["detail"]


def test_limits_come_from_environment_with_zero_meaning_unlimited(monkeypatch):
    for name in ("RPM", "TPM", "RPD", "MAX_CALLS"):
        monkeypatch.delenv(f"AGENTRAG_GEMINI_{name}", raising=False)
    assert limits_from_env() == Limits(rpm=10, tpm=200_000, rpd=None, max_calls=None)
    monkeypatch.setenv("AGENTRAG_GEMINI_RPM", "15")
    monkeypatch.setenv("AGENTRAG_GEMINI_TPM", "0")
    monkeypatch.setenv("AGENTRAG_GEMINI_RPD", "1500")
    monkeypatch.setenv("AGENTRAG_GEMINI_MAX_CALLS", "300")
    assert limits_from_env() == Limits(rpm=15, tpm=None, rpd=1500, max_calls=300)


def test_a_transient_windows_style_replace_failure_is_retried_and_never_raises(tmp_path, monkeypatch):
    import os

    real_replace, calls = os.replace, {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(5, "Access is denied")
        return real_replace(src, dst)

    monkeypatch.setattr("agentrag.verify.ratelimit.os.replace", flaky_replace)
    monkeypatch.setattr("agentrag.verify.ratelimit.time.sleep", lambda s: None)
    limiter, _ = _limiter(Limits(rpm=None, tpm=None), tmp_path)
    limiter.acquire()
    assert calls["n"] == 3
    reloaded = RateLimiter(Limits(rpm=None, tpm=None), tmp_path / "ledger.json", day_fn=lambda: "d")
    assert reloaded.ledger.lifetime()["calls"] == 1  # the retried write did land


def test_a_ledger_that_cannot_be_written_at_all_keeps_counting_in_memory(tmp_path, monkeypatch):
    def always_fail(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr("agentrag.verify.ratelimit.os.replace", always_fail)
    monkeypatch.setattr("agentrag.verify.ratelimit.time.sleep", lambda s: None)
    monkeypatch.setattr("pathlib.Path.write_text", lambda self, *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    limiter, _ = _limiter(Limits(rpm=None, tpm=None), tmp_path)
    limiter.acquire()
    limiter.settle(0.01)
    assert limiter.ledger.lifetime() == {"calls": 1, "usd": 0.01}
