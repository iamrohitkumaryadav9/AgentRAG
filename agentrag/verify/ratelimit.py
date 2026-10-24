"""Client-side rate limiting and budget accounting for the Gemini API.

Limits are *per project* and not discoverable from the public docs (they defer
to the AI Studio dashboard), so nothing here hard-codes them: set
``AGENTRAG_GEMINI_RPM`` / ``_TPM`` / ``_RPD`` / ``_MAX_CALLS`` to what your
project shows. Unset values mean "no client-side limit of that kind" except RPM
and TPM, which default to conservative values. The server's 429 response is
still honoured (``penalize`` / ``exhaust_today``), so a wrong guess costs time,
not correctness.

A persistent ledger (keyed by the Pacific date, when Google's daily quotas
reset) lets a long experiment stop cleanly when the day's budget is spent and
resume later without double-counting. It also keeps *lifetime* totals of
requests and spend for the hard budget in ``budget.py``. The ledger file is
named after a fingerprint of the API key, so a new key (a new project, new
quota, new credits) starts with a clean ledger instead of inheriting another
project's exhausted quota.

Everything here is thread-safe: concurrent workers share one limiter.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .budget import Budget, BudgetExceeded

logger = logging.getLogger(__name__)

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
    max_calls: int | None = None  # hard cap for this process, independent of the ledger


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


def key_fingerprint(api_key: str) -> str:
    """Short, non-reversible tag identifying which key a ledger belongs to."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:8]


def ledger_path_for_key(api_key: str, directory: Path | str = ".verify_state") -> Path:
    return Path(directory) / f"ledger-{key_fingerprint(api_key)}.json"


class Ledger:
    """Per-day request counts plus lifetime totals, persisted so a stopped run resumes correctly."""

    def __init__(self, path: Path | str | None, day_fn: Callable[[], str]) -> None:
        self.path = Path(path) if path is not None else None
        self._day_fn = day_fn
        self._lock = threading.RLock()
        self._data: dict = {"counts": {}, "exhausted": {}, "last_429": None, "lifetime": {"calls": 0, "usd": 0.0}}
        if self.path is not None and self.path.exists():
            try:
                self._data.update(json.loads(self.path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
            self._data.setdefault("lifetime", {"calls": 0, "usd": 0.0})

    def _save(self) -> None:
        """Persist the ledger. Never raises: the in-memory totals stay authoritative if the disk misbehaves.

        On Windows ``os.replace`` can fail transiently while an indexer or antivirus holds the destination,
        so it is retried; as a last resort the file is written in place. A bookkeeping failure must never be
        able to discard the response to a request that has already been paid for.
        """
        if self.path is None:
            return
        payload = json.dumps(self._data, indent=1, sort_keys=True)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(8):
                tmp = self.path.with_name(f"{self.path.name}.{threading.get_ident()}.tmp")
                try:
                    tmp.write_text(payload, encoding="utf-8")
                    os.replace(tmp, self.path)
                    return
                except OSError:
                    time.sleep(0.05 * (attempt + 1))
            self.path.write_text(payload, encoding="utf-8")
        except OSError as err:
            logger.error("could not save the spend ledger %s (%s); continuing with in-memory totals", self.path, err)

    def count_today(self) -> int:
        with self._lock:
            return int(self._data["counts"].get(self._day_fn(), 0))

    def increment(self) -> None:
        with self._lock:
            day = self._day_fn()
            self._data["counts"][day] = self.count_today() + 1
            self._data["lifetime"]["calls"] += 1
            self._save()

    def add_spend(self, usd: float) -> None:
        with self._lock:
            self._data["lifetime"]["usd"] += usd
            self._save()

    def lifetime(self) -> dict:
        with self._lock:
            return dict(self._data["lifetime"])

    def exhausted_today(self) -> bool:
        with self._lock:
            return bool(self._data["exhausted"].get(self._day_fn(), False))

    def mark_exhausted(self, detail: str = "") -> None:
        with self._lock:
            self._data["exhausted"][self._day_fn()] = True
            self._data["last_429"] = {"day": self._day_fn(), "detail": detail[:500]}
            self._save()


class RateLimiter:
    def __init__(
        self,
        limits: Limits | None = None,
        ledger_path: Path | str | None = None,
        *,
        budget: Budget | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        day_fn: Callable[[], str] | None = None,
    ) -> None:
        self.limits = limits or limits_from_env()
        self.budget = budget
        self._clock, self._sleep = clock, sleep
        self._window: deque[list] = deque()  # [timestamp, tokens]
        self._cooldown_until = 0.0
        self._reserved_usd = 0.0
        self._lock = threading.RLock()
        self.calls_this_run = 0
        self.ledger = Ledger(ledger_path, day_fn or (lambda: pacific_day(wall())))

    # -- public API ---------------------------------------------------------------

    def acquire(self, est_tokens: int = 0) -> list:
        """Block until a request is allowed, then reserve it. Raises when a hard limit is hit.

        Returns a handle for ``record_tokens``. Every successful ``acquire`` must be paired with exactly one
        ``settle`` call so the spend reservation is released.
        """
        lim = self.limits
        if lim.tpm is not None and est_tokens > lim.tpm:
            raise ValueError(f"a single request (~{est_tokens} tokens) exceeds the TPM limit ({lim.tpm})")

        while True:
            with self._lock:
                self._check_hard_limits()
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
                    entry = [self._clock(), est_tokens]
                    self._window.append(entry)
                    self.calls_this_run += 1
                    self.ledger.increment()
                    if self.budget is not None:
                        self._reserved_usd += self.budget.tripwire_usd  # worst case for one call
                    return entry
            self._sleep(wait + 0.01)  # never sleep while holding the lock

    def record_tokens(self, actual_tokens: int, entry: list | None = None) -> None:
        """Replace a request's token estimate with its real usage (default: the most recent request)."""
        with self._lock:
            target = entry if entry is not None else (self._window[-1] if self._window else None)
            if target is not None:
                target[1] = actual_tokens

    def settle(self, actual_usd: float = 0.0) -> None:
        """Release one request's spend reservation and charge what it actually cost."""
        with self._lock:
            if self.budget is not None:
                self._reserved_usd = max(0.0, self._reserved_usd - self.budget.tripwire_usd)
            if actual_usd:
                self.ledger.add_spend(actual_usd)

    def penalize(self, retry_after_seconds: float) -> None:
        """The server said slow down: no request until the retry delay has passed."""
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, self._clock() + max(0.0, retry_after_seconds))

    def exhaust_today(self, detail: str = "") -> None:
        self.ledger.mark_exhausted(detail)

    def spend_summary(self) -> dict:
        life = self.ledger.lifetime()
        out = {"lifetime_calls": life["calls"], "lifetime_usd": life["usd"]}
        budget = self.budget
        if budget is not None:
            spent_inr = budget.inr(life["usd"])
            out.update(
                lifetime_inr=spent_inr,
                remaining_inr=None if budget.max_inr is None else budget.max_inr - spent_inr,
                remaining_calls=None if budget.max_calls is None else budget.max_calls - life["calls"],
            )
        return out

    # -- internals ----------------------------------------------------------------

    def _check_hard_limits(self) -> None:
        lim, budget = self.limits, self.budget
        if lim.max_calls is not None and self.calls_this_run >= lim.max_calls:
            raise DailyBudgetExhausted(f"hard call cap reached ({lim.max_calls} calls this run)")
        if self.ledger.exhausted_today():
            raise DailyBudgetExhausted("the server reported today's quota as exhausted")
        if lim.rpd is not None and self.ledger.count_today() >= lim.rpd:
            raise DailyBudgetExhausted(f"daily request budget reached ({lim.rpd} requests today)")
        if budget is not None:
            life = self.ledger.lifetime()
            if budget.max_calls is not None and life["calls"] >= budget.max_calls:
                raise BudgetExceeded(f"call cap reached ({life['calls']} of {budget.max_calls} requests)")
            if budget.max_usd is not None and life["usd"] + self._reserved_usd + budget.tripwire_usd > budget.max_usd:
                raise BudgetExceeded(
                    f"spend cap reached (Rs {budget.inr(life['usd']):.2f} spent, Rs {budget.max_inr:.0f} cap)"
                )
