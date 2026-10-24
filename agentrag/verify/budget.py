"""Hard spending limits for API experiments, enforced in code.

Rules (defaults are the ones set for the SciFact verification study):

* stop all API calls at ``max_inr`` rupees of total spend or ``max_calls`` requests, whichever comes first;
* optional *phase caps* (ledger-backed, see ``RateLimiter.start_phase``) bound the spend of one stage of the
  study, for example "finish the train arms for at most Rs 40", and survive restarts;
* spend is tracked from token counts at the model's list price and persisted in the ledger, so a restarted
  run continues from where the last one stopped (it cannot "forget" what was spent);
* every in-flight request *reserves* ``reserve_usd`` before it is sent, so N concurrent workers can never
  overshoot a cap by more than those reservations;
* tripwire: a single call that costs more than ``tripwire_usd`` stops everything, after the response has been
  cached so it is never re-requested. Thinking tokens are logged and reported per arm but are not a stop
  condition by default (``max_thinking_tokens=None``): the model cannot disable thinking, and a retry that asks it
  to reconsider can legitimately use a few hundred to a couple of thousand thought tokens (a fraction of a rupee).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# 600 rupees ~= 6.8 US dollars, the conversion used when the cap was set.
INR_PER_USD = 600 / 6.8


class BudgetExceeded(RuntimeError):
    """The call cap, the spend cap or a phase cap was reached; no further request may be sent."""


class TripwireError(RuntimeError):
    """A single call cost more than the tripwire allows. Stop and report."""


@dataclass(frozen=True)
class Budget:
    max_calls: int | None = 2000
    max_inr: float | None = 600.0
    inr_per_usd: float = INR_PER_USD
    tripwire_usd: float = 0.02  # per-call stop; ordinary calls cost ~$0.002, retries that think up to ~$0.006
    reserve_usd: float = 0.006  # what an in-flight request is assumed to cost until it settles
    max_thinking_tokens: int | None = None  # None: thinking is logged, never a stop condition

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
    thinking = _env_float("AGENTRAG_BUDGET_MAX_THINKING_TOKENS", None)
    return Budget(
        max_calls=None if calls is None else int(calls),
        max_inr=_env_float("AGENTRAG_BUDGET_MAX_INR", 600.0),
        inr_per_usd=_env_float("AGENTRAG_BUDGET_INR_PER_USD", INR_PER_USD) or INR_PER_USD,
        tripwire_usd=_env_float("AGENTRAG_BUDGET_TRIPWIRE_USD", 0.02) or 0.02,
        reserve_usd=_env_float("AGENTRAG_BUDGET_RESERVE_USD", 0.006) or 0.006,
        max_thinking_tokens=None if thinking is None else int(thinking),
    )
