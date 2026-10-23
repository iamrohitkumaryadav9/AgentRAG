"""Non-LLM polarity checking with an NLI model (DeBERTa trained on MNLI + FEVER + ANLI).

The model's label order is *read from its config and asserted*, not assumed, so
a different checkpoint with a different order fails loudly instead of silently
swapping "entails" and "contradicts" (which would invert the critic).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from .common import RelationResult, Usage
from .schemas import Relation

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
EXPECTED_LABELS = {"entailment", "neutral", "contradiction"}

Probs = Mapping[str, float]  # keys: entailment / neutral / contradiction


def decide_relation(probs: Probs, tau: float) -> Relation:
    """Entails / contradicts only when that class wins *and* clears ``tau``; otherwise neutral."""
    entail, contra = probs["entailment"], probs["contradiction"]
    if entail >= contra and entail >= probs["neutral"] and entail >= tau:
        return Relation.ENTAILS
    if contra > entail and contra >= probs["neutral"] and contra >= tau:
        return Relation.CONTRADICTS
    return Relation.NEUTRAL


class NliScorer:
    """A transformers NLI model: ``score(premise, hypothesis)`` returns class probabilities."""

    def __init__(self, model_name: str = NLI_MODEL, max_length: int = 512) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        labels = {int(i): str(name).lower() for i, name in self.model.config.id2label.items()}
        if set(labels.values()) != EXPECTED_LABELS:
            raise RuntimeError(f"unexpected NLI labels {labels}; refusing to guess their order")
        self.id2label, self.max_length, self.model_name = labels, max_length, model_name

    def score_many(self, pairs: Sequence[tuple[str, str]], batch_size: int = 16) -> list[dict[str, float]]:
        torch, out = self._torch, []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            enc = self.tokenizer(
                [p for p, _ in batch], [h for _, h in batch],
                truncation="only_first", max_length=self.max_length, padding=True, return_tensors="pt",
            )
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, dim=-1)
            out += [{self.id2label[i]: float(row[i]) for i in range(len(row))} for row in probs]
        return out

    def score(self, premise: str, hypothesis: str) -> dict[str, float]:
        return self.score_many([(premise, hypothesis)])[0]


class NliPolarityChecker:
    """Blind polarity check without an LLM: does the cited text entail / contradict the claim?"""

    def __init__(self, scorer: Callable[[str, str], Probs], tau: float, name: str = "nli") -> None:
        self.scorer, self.tau, self.name = scorer, tau, name

    def relation(self, claim: str, sentences: Sequence[str]) -> RelationResult:
        probs = self.scorer(" ".join(sentences), claim)
        return RelationResult(decide_relation(probs, self.tau), Usage(model_calls=1), dict(probs))
