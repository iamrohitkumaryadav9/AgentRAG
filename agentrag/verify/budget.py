"""Hard spending limits for API experiments, enforced in code.

Rules (defaults are the ones set for the SciFact verification study):

* stop all API calls at ``max_inr`` rupees of total spend or ``max_calls`` requests, whichever comes first;
* spend is tracked from token counts at the model's list price and persisted in the ledger, so a restarted
  run continues from where the last one stopped (it cannot "forget" what was spent);
* every in-flight request *reserves* the worst-case cost of one call before it is sent, so N concurrent
  workers can never overshoot the cap by more than the reservations already accounted for;
* tripwire: a single call that costs more than ``tripwire_usd`` (2x the ~$0.002 estimate) or that reports any
  thinking tokens stops everything, after the response has been cached so it is never re-requested.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# 600 rupees ~= 6.8 US dollars, the conversion used when the cap was set.
INR_PER_USD = 600 / 6.8


class BudgetExceeded(RuntimeError):
    """The call cap or the spend cap was reached; no further request may be sent."""


class TripwireError(RuntimeError):
    """A call cost more than the tripwire allows, or thinking tokens appeared. Stop and report."""


@dataclass(frozen=True)
class Budget:
    max_calls: int | None = 2000
    max_inr: float | None = 600.0
    inr_per_usd: float = INR_PER_USD
    tripwire_usd: float = 0.004  # 2x the ~$0.002 per-call estimate
    allow_thinking_tokens: bool = False

    @property
    def max_usd(self) -> float | None:
        return None if self.max_inr is None else self.max_inr / self.inr_per_usd

    def inr(self, usd: float) -> float:
        return usd * self.inr_per_usd


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = float(raw)
    return None if value <= 0 else value


def budget_from_env() -> Budget:
    calls = _env_float("AGENTRAG_BUDGET_MAX_CALLS", 2000)
    return Budget(
        max_calls=None if calls is None else int(calls),
        max_inr=_env_float("AGENTRAG_BUDGET_MAX_INR", 600.0),
        inr_per_usd=_env_float("AGENTRAG_BUDGET_INR_PER_USD", INR_PER_USD) or INR_PER_USD,
        tripwire_usd=_env_float("AGENTRAG_BUDGET_TRIPWIRE_USD", 0.004) or 0.004,
    )
