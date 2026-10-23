"""A cached, rate-limited, retrying Gemini client with structured output.

``GeminiClient`` owns the policy (cache first, limiter before every request,
backoff on transient errors, one JSON repair attempt, safety blocks recorded as
results) and talks to the network only through a ``Transport``. Tests use a fake
transport; ``GenaiTransport`` adapts the google-genai SDK and is the only code
that imports it.

Thinking: gemini-3.8-flash cannot be switched off. ``thinking_budget=0`` is
accepted but ignored and ``MINIMAL`` is rejected (verified against the live
API), so the lowest available level is ``low``. Thought tokens are billed as
output tokens and are tracked separately.
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from pydantic import BaseModel, ValidationError

from .cache import CachedResponse, CacheMiss, ResponseCache, make_key
from .ratelimit import DailyBudgetExhausted, RateLimiter
from .schemas import parse_model_json

# USD per 1M tokens (input, output). Thinking tokens are billed as output. Paid-tier list prices for
# gemini-3.8-flash through 2026-12-31; the free tier charges nothing, so this is a paid-equivalent figure.
PRICES_USD_PER_M: dict[str, tuple[float, float]] = {"gemini-3.8-flash": (0.75, 3.75)}

THINKING_LEVELS = ("default", "low", "medium", "high")


class TransientError(RuntimeError):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RateLimitError(RuntimeError):
    def __init__(self, message: str, retry_after: float | None = None, daily: bool = False) -> None:
        super().__init__(message)
        self.retry_after, self.daily = retry_after, daily


class InvalidModelOutput(RuntimeError):
    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


class BudgetExceeded(RuntimeError):
    """The configured paid-equivalent spend cap was reached."""


@dataclass
class RawResponse:
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    finish_reason: str = ""
    model_version: str = ""
    blocked: bool = False


class Transport(Protocol):
    def call(self, prompt: str, schema: type[BaseModel], temperature: float, thinking: str) -> RawResponse: ...


@dataclass
class CallResult:
    parsed: BaseModel | None  # None when the response was blocked
    response: CachedResponse
    from_cache: bool
    api_calls: int  # requests actually sent for this result (0 on a cache hit; 2 if a repair was needed)

    @property
    def blocked(self) -> bool:
        return self.response.blocked


def cost_usd(model: str, prompt_tokens: int, output_tokens: int, thought_tokens: int) -> float:
    price_in, price_out = PRICES_USD_PER_M.get(model, (0.0, 0.0))
    return (prompt_tokens * price_in + (output_tokens + thought_tokens) * price_out) / 1_000_000


# -- error translation (kept SDK-free so it can be unit tested) ---------------------------------


def _walk(node):
    """Yield every dict nested anywhere inside ``node``."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list | tuple):
        for item in node:
            yield from _walk(item)


def _parse_seconds(value) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.endswith("s"):
        try:
            return float(value[:-1])
        except ValueError:
            return None
    return None


def translate_error(code: int | None, status: str | None, message: str, details) -> Exception | None:
    """Map an API error to RateLimitError / TransientError, or None if it should propagate as-is."""
    if code == 429 or status == "RESOURCE_EXHAUSTED":
        retry_after, quota_ids = None, []
        for entry in _walk(details):
            if "retryDelay" in entry:
                retry_after = _parse_seconds(entry["retryDelay"])
            if "quotaId" in entry:
                quota_ids.append(str(entry["quotaId"]))
        daily = any("PerDay" in q for q in quota_ids)
        return RateLimitError(message, retry_after=retry_after, daily=daily)
    if code in (500, 502, 503, 504) or status in ("UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"):
        return TransientError(message)
    return None


class GenaiTransport:
    """The google-genai SDK adapter. Imports the SDK lazily; never logs request configuration."""

    def __init__(self, model: str, api_key: str, timeout_seconds: float = 120.0) -> None:
        from google import genai
        from google.genai import types

        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        self.model = model
        self._types = types
        options = types.HttpOptions(timeout=int(timeout_seconds * 1000))
        self._client = genai.Client(api_key=api_key, http_options=options)

    def call(self, prompt: str, schema: type[BaseModel], temperature: float, thinking: str) -> RawResponse:
        from google.genai import errors

        types = self._types
        config: dict = {"temperature": temperature, "response_mime_type": "application/json", "response_schema": schema}
        if thinking != "default":
            config["thinking_config"] = types.ThinkingConfig(thinking_level=thinking.upper())
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=prompt, config=types.GenerateContentConfig(**config)
            )
        except errors.APIError as exc:
            translated = translate_error(exc.code, exc.status, str(exc.message), exc.details)
            raise (translated or exc) from exc
        except OSError as exc:  # connection resets, timeouts surfaced by the HTTP layer
            raise TransientError(f"{type(exc).__name__}: {exc}") from exc

        usage = response.usage_metadata
        feedback = getattr(response, "prompt_feedback", None)
        candidates = getattr(response, "candidates", None) or []
        finish = str(getattr(candidates[0], "finish_reason", "")) if candidates else ""
        blocked = bool(getattr(feedback, "block_reason", None)) or not candidates or "SAFETY" in finish
        try:
            text = response.text or ""
        except ValueError:  # the SDK raises when there is no text part (e.g. a safety stop)
            text, blocked = "", True
        return RawResponse(
            text=text,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            thought_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
            finish_reason=finish,
            model_version=getattr(response, "model_version", "") or "",
            blocked=blocked and not text,
        )


class GeminiClient:
    def __init__(
        self,
        model: str,
        cache: ResponseCache,
        limiter: RateLimiter,
        transport: Transport | None = None,
        *,
        thinking: str = "low",
        temperature: float = 0.0,
        max_retries: int = 5,
        base_delay: float = 2.0,
        max_usd: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        if thinking not in THINKING_LEVELS:
            raise ValueError(f"thinking must be one of {THINKING_LEVELS}, got {thinking!r}")
        self.model, self.cache, self.limiter, self.transport = model, cache, limiter, transport
        self.thinking, self.temperature = thinking, temperature
        self.max_retries, self.base_delay, self.max_usd = max_retries, base_delay, max_usd
        self._sleep, self._rng = sleep, rng or random.Random(0)
        self.spent_usd = 0.0
        self.api_calls = 0

    # -- public -----------------------------------------------------------------

    def generate(self, prompt: str, schema: type[BaseModel], *, prompt_version: str, label: str) -> CallResult:
        """Return the parsed response, from cache when this exact request was answered before."""
        first, cached, calls = self._request(prompt, schema, prompt_version, label, attempt=0)
        if first.blocked:
            return CallResult(None, first, cached, calls)
        try:
            return CallResult(parse_model_json(schema, first.text), first, cached, calls)
        except ValidationError as err:
            problem = str(err).splitlines()[0][:200]

        repair = (
            f"{prompt}\n\nYour previous reply was not valid JSON for the required schema ({problem}). "
            "Reply again with ONLY the JSON object."
        )
        second, cached2, calls2 = self._request(repair, schema, prompt_version, label + ":repair", attempt=1)
        cached, calls = cached and cached2, calls + calls2
        if second.blocked:
            return CallResult(None, second, cached, calls)
        try:
            return CallResult(parse_model_json(schema, second.text), second, cached, calls)
        except ValidationError as err:
            detail = str(err).splitlines()[0]
            raise InvalidModelOutput(f"invalid output after one repair attempt: {detail}", second.text) from err

    # -- internals ----------------------------------------------------------------

    def _request(self, prompt, schema, prompt_version, label, attempt) -> tuple[CachedResponse, bool, int]:
        """Return (response, served_from_cache, requests_sent)."""
        key = make_key(
            model=self.model, prompt_version=prompt_version, temperature=self.temperature,
            thinking=self.thinking, schema=schema.__name__, prompt=prompt, attempt=attempt,
        )
        hit = self.cache.get(key)
        if hit is not None:
            return hit, True, 0
        if self.cache.offline:
            raise CacheMiss(f"offline mode and no cached response for {label} (key {key[:12]})")
        if self.transport is None:
            raise RuntimeError("no transport configured and the request is not cached")
        if self.max_usd is not None and self.spent_usd >= self.max_usd:
            raise BudgetExceeded(f"paid-equivalent spend cap reached (${self.spent_usd:.2f} of ${self.max_usd:.2f})")

        raw, latency_ms, sent = self._send(prompt, schema)
        self.api_calls += sent
        self.spent_usd += cost_usd(self.model, raw.prompt_tokens, raw.output_tokens, raw.thought_tokens)
        response = CachedResponse(
            key=key, label=label, text=raw.text, prompt_tokens=raw.prompt_tokens, output_tokens=raw.output_tokens,
            thought_tokens=raw.thought_tokens, latency_ms=round(latency_ms, 1), model=self.model,
            model_version=raw.model_version, finish_reason=raw.finish_reason, blocked=raw.blocked,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.cache.put(response)
        return response, False, sent

    def _send(self, prompt: str, schema: type[BaseModel]) -> tuple[RawResponse, float, int]:
        delay, sent = self.base_delay, 0
        estimate = len(prompt) // 4 + 600  # prompt plus room for output and thinking
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire(estimate)
            sent += 1
            start = time.perf_counter()
            try:
                raw = self.transport.call(prompt, schema, self.temperature, self.thinking)
            except RateLimitError as err:
                if err.daily:
                    self.limiter.exhaust_today(str(err))
                    raise DailyBudgetExhausted(f"server reports the daily quota is spent: {str(err)[:200]}") from err
                self.limiter.penalize(err.retry_after if err.retry_after is not None else delay)
                if attempt == self.max_retries:
                    raise
                delay = min(delay * 2, 60.0)
            except TransientError as err:
                if attempt == self.max_retries:
                    raise
                self._sleep(err.retry_after if err.retry_after is not None else delay * self._rng.uniform(0.5, 1.5))
                delay = min(delay * 2, 60.0)
            else:
                self.limiter.record_tokens(raw.prompt_tokens + raw.output_tokens + raw.thought_tokens)
                return raw, (time.perf_counter() - start) * 1000, sent
        raise AssertionError("unreachable")  # pragma: no cover
