"""The NOT_ENOUGH_INFO recheck: catch a judge that declined although a retrieved sentence settles the claim.

On the 60 train claims every one of the judge's errors was a false NOT_ENOUGH_INFO, and none of them cites a
sentence, so neither the citation rules nor the blind polarity check (both of which need a cited sentence) can see
them. This rule works from the other side: score every retrieved sentence against the claim with the NLI model and,
if some sentence entails or contradicts it with probability above a threshold, flag the NOT_ENOUGH_INFO verdict and
let the judge retry with feedback naming that sentence.

The strongest sentence is a property of (claim, retrieved documents) alone. The threshold only decides whether it
counts, so the same retry prompt serves every threshold at or above the lowest one tried, which lets the threshold
be tuned on stored retry outcomes without further API calls.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .data import Doc
from .schemas import Relation

Probs = Mapping[str, float]


@dataclass(frozen=True)
class SentenceHit:
    doc_id: str
    index: int
    text: str
    relation: Relation  # ENTAILS or CONTRADICTS
    prob: float


class NeiRecheck:
    def __init__(self, score_many: Callable[[Sequence[tuple[str, str]]], list[Probs]], tau: float) -> None:
        self.score_many, self.tau = score_many, tau
        self._memo: dict[tuple, SentenceHit | None] = {}
        self._lock = threading.Lock()

    def strongest(self, claim: str, docs: Sequence[Doc]) -> SentenceHit | None:
        """The retrieved sentence most strongly entailing or contradicting the claim (None without sentences)."""
        key = (claim, tuple(d.id for d in docs))
        with self._lock:
            if key in self._memo:
                return self._memo[key]
            located = [(d.id, i, s) for d in docs for i, s in enumerate(d.sentences)]
            best: SentenceHit | None = None
            if located:
                probs = self.score_many([(s, claim) for _, _, s in located])
                for (doc_id, index, text), p in zip(located, probs, strict=True):
                    entail, contra = p["entailment"], p["contradiction"]
                    prob = max(entail, contra)
                    if best is None or prob > best.prob:
                        relation = Relation.ENTAILS if entail >= contra else Relation.CONTRADICTS
                        best = SentenceHit(doc_id, index, text, relation, float(prob))
            self._memo[key] = best
            return best

    def find(self, claim: str, docs: Sequence[Doc]) -> SentenceHit | None:
        hit = self.strongest(claim, docs)
        return hit if hit is not None and hit.prob > self.tau else None
