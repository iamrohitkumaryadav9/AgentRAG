"""Client-side rate limiting for the Gemini API, configurable for any tier.

Limits are *per project* and not discoverable from the public docs (they defer
to the AI Studio dashboard), so nothing here hard-codes them: set
``AGENTRAG_GEMINI_RPM`` / ``_TPM`` / ``_RPD`` / ``_MAX_CALLS`` to what your
project shows. Unset values mean "no client-side limit of that kind" except RPM
and TPM, which default to conservative values. The server's 429 response is
still honoured (``penalize`` / ``exhaust_today``), so a wrong guess costs time,
not correctness.

A persistent daily ledger (keyed by the Pacific date, when Google's daily
quotas reset) lets a long experiment stop cleanly when the day's budget is
spent and resume the next day without double-counting.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_RPM = 10
DEFAULT_TPM = 200_000
WINDOW_SECONDS = 60.0


class DailyBudgetExhausted(RuntimeError):
    """The day's request budget (client-side RPD, a hard call cap, or a server daily 429) is spent."""


@dataclass(frozen=True)
class Limits:
    rpm: int | None = DEFAULT_RPM
    tpm: int | None = DEFAULT_TPM
    rpd: int | None = None
    max_calls: int | None = None  # hard cap for this process, independent of the daily ledger


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    return None if value <= 0 else value  # 0 / negative means "no limit"


def limits_from_env() -> Limits:
    return Limits(
        rpm=_env_int("AGENTRAG_GEMINI_RPM", DEFAULT_RPM),
        tpm=_env_int("AGENTRAG_GEMINI_TPM", DEFAULT_TPM),
        rpd=_env_int("AGENTRAG_GEMINI_RPD", None),
        max_calls=_env_int("AGENTRAG_GEMINI_MAX_CALLS", None),
    )


def pacific_day(epoch_seconds: float) -> str:
    """Calendar date in US Pacific time, the clock Google's daily quotas reset on."""
    from zoneinfo import ZoneInfo

    return datetime.fromtimestamp(epoch_seconds, ZoneInfo("America/Los_Angeles")).date().isoformat()


class Ledger:
    """Per-day request counts, persisted so a stopped run resumes with the right budget."""

    def __init__(self, path: Path | str | None, day_fn: Callable[[], str]) -> None:
        self.path = Path(path) if path is not None else None
        self._day_fn = day_fn
        self._data: dict = {"counts": {}, "exhausted": {}, "last_429": None}
        if self.path is not None and self.path.exists():
            try:
                self._data.update(json.loads(self.path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def count_today(self) -> int:
        return int(self._data["counts"].get(self._day_fn(), 0))

    def increment(self) -> None:
        day = self._day_fn()
        self._data["counts"][day] = self.count_today() + 1
        self._save()

    def exhausted_today(self) -> bool:
        return bool(self._data["exhausted"].get(self._day_fn(), False))

    def mark_exhausted(self, detail: str = "") -> None:
        self._data["exhausted"][self._day_fn()] = True
        self._data["last_429"] = {"day": self._day_fn(), "detail": detail[:500]}
        self._save()


class RateLimiter:
    def __init__(
        self,
        limits: Limits | None = None,
        ledger_path: Path | str | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        day_fn: Callable[[], str] | None = None,
    ) -> None:
        self.limits = limits or limits_from_env()
        self._clock, self._sleep = clock, sleep
        self._window: deque[list] = deque()  # [timestamp, tokens]
        self._cooldown_until = 0.0
        self.calls_this_run = 0
        self.ledger = Ledger(ledger_path, day_fn or (lambda: pacific_day(wall())))

    # -- public API ---------------------------------------------------------------

    def acquire(self, est_tokens: int = 0) -> None:
        """Block until a request is allowed, then reserve it. Raises DailyBudgetExhausted when spent."""
        lim = self.limits
        if lim.tpm is not None and est_tokens > lim.tpm:
            raise ValueError(f"a single request (~{est_tokens} tokens) exceeds the TPM limit ({lim.tpm})")
        if lim.max_calls is not None and self.calls_this_run >= lim.max_calls:
            raise DailyBudgetExhausted(f"hard call cap reached ({lim.max_calls} calls this run)")
        if self.ledger.exhausted_today():
            raise DailyBudgetExhausted("the server reported today's quota as exhausted")
        if lim.rpd is not None and self.ledger.count_today() >= lim.rpd:
            raise DailyBudgetExhausted(f"daily request budget reached ({lim.rpd} requests today)")

        while True:
            now = self._clock()
            while self._window and now - self._window[0][0] >= WINDOW_SECONDS:
                self._window.popleft()
            wait = max(0.0, self._cooldown_until - now)
            if lim.rpm is not None and len(self._window) >= lim.rpm:
                wait = max(wait, self._window[0][0] + WINDOW_SECONDS - now)
            if lim.tpm is not None:
                used = sum(tokens for _, tokens in self._window)
                if used + est_tokens > lim.tpm:
                    freed, release_at = 0, now
                    for stamp, tokens in self._window:
                        freed += tokens
                        release_at = stamp + WINDOW_SECONDS
                        if used - freed + est_tokens <= lim.tpm:
                            break
                    wait = max(wait, release_at - now)
            if wait <= 0:
                break
            self._sleep(wait + 0.01)

        self._window.append([self._clock(), est_tokens])
        self.calls_this_run += 1
        self.ledger.increment()

    def record_tokens(self, actual_tokens: int) -> None:
        """Replace the estimate for the most recent request with the real usage."""
        if self._window:
            self._window[-1][1] = actual_tokens

    def penalize(self, retry_after_seconds: float) -> None:
        """The server said slow down: no request until the retry delay has passed."""
        self._cooldown_until = max(self._cooldown_until, self._clock() + max(0.0, retry_after_seconds))

    def exhaust_today(self, detail: str = "") -> None:
        self.ledger.mark_exhausted(detail)
