"""Append-only, content-addressed cache of model responses.

Every call is keyed by everything that can change its answer (model, prompt
version, temperature, thinking level, schema, the rendered prompt, attempt),
so an identical request is answered from disk: reruns of a finished
experiment cost nothing and are bit-for-bit reproducible.

Only the response and its usage are stored. Request headers, API keys and
client configuration never enter this module, and ``find_secrets`` lets a
test (and the pre-commit check) prove a cache file is clean before it is
committed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path


class CacheMiss(RuntimeError):
    """Raised in offline mode when a request has no cached response."""


@dataclass
class CachedResponse:
    key: str
    label: str  # human-readable purpose, e.g. "judge:v1"
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    model_version: str = ""
    finish_reason: str = ""
    blocked: bool = False
    created_at: str = ""


FIELDS = tuple(CachedResponse.__dataclass_fields__)  # the only things ever written to disk


def make_key(
    *, model: str, prompt_version: str, temperature: float, thinking: str, schema: str, prompt: str, attempt: int = 0
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt_version": prompt_version,
            "temperature": temperature,
            "thinking": thinking,
            "schema": schema,
            "prompt": prompt,
            "attempt": attempt,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: Path | str | None, offline: bool = False) -> None:
        self.path = Path(path) if path is not None else None
        self.offline = offline
        self._entries: dict[str, CachedResponse] = {}
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    self._entries[row["key"]] = CachedResponse(**{k: row[k] for k in FIELDS if k in row})
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue  # a partial last line from an interrupted write; the next put re-adds it

    def get(self, key: str) -> CachedResponse | None:
        return self._entries.get(key)

    def put(self, response: CachedResponse) -> None:
        self._entries[response.key] = response
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(response), ensure_ascii=False, sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries


# Shapes of credentials and request metadata that must never appear in a committed cache.
_SECRET_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"AQ\.[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)x-goog-api-key"),
    re.compile(r"(?i)authorization\s*[:=]"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{16,}"),
    re.compile(r"(?i)\"api_key\"\s*:"),
)


def find_secrets(text: str) -> list[str]:
    """Return the patterns (as strings) that match, never the matched text itself."""
    return [p.pattern for p in _SECRET_PATTERNS if p.search(text)]
