"""Versioned prompts for the judge and the blind polarity critic.

Templates live in registries keyed by version, so a revised prompt gets a new
version while the old one stays renderable (earlier cached results remain
reproducible and a revision is a reviewable diff, not an overwrite). The version
string is part of every cache key, and ``config_fingerprint`` hashes the full
text of the selected versions plus the run settings, so "freezing the prompts"
before the final run is a checkable statement: the fingerprint printed by the
run script must match the one recorded when the freeze was committed.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from .data import Doc

JUDGE_VERSION = "judge-v1"  # the version a run uses unless told otherwise
RELATION_VERSION = "relation-v1"

JUDGE_TEMPLATES: dict[str, str] = {
    "judge-v1": """You are verifying a scientific claim against retrieved paper abstracts.

Choose exactly one verdict:
- SUPPORT: the abstracts contain sentences that, taken together, directly support the claim.
- CONTRADICT: the abstracts contain sentences that directly contradict the claim.
- NOT_ENOUGH_INFO: no sentence settles the claim either way. A related topic is not enough.

Rules:
- Judge only from the text below, never from outside knowledge.
- Direction and negation matter: increase versus decrease, more versus less, causes versus does not cause. \
A claim and its mirror image can have opposite verdicts.
- Cite the sentences your verdict rests on as {{"doc_id": "<id>", "sentence_ids": [<numbers shown in brackets>]}}. \
Cite nothing for NOT_ENOUGH_INFO.
- Give a one-sentence rationale.

Claim: {claim}

Abstracts:
{abstracts}
{feedback}
Respond with JSON only.""",
}

FEEDBACK_TEMPLATE = (
    "\nA reviewer checked your previous answer and found a problem: {problem}\n"
    "Re-read the claim and the cited sentences, then answer again.\n"
)

RELATION_TEMPLATES: dict[str, str] = {
    "relation-v1": """Decide how the evidence relates to the claim.

- "entails": taken at face value, the evidence makes the claim true.
- "contradicts": taken at face value, the evidence makes the claim false.
- "neutral": the evidence does not settle the claim either way.

Direction and negation matter: increase versus decrease, more versus less, causes versus does not cause.

Claim: {claim}

Evidence:
{evidence}

Respond with JSON only.""",
}


def render_abstracts(docs: Sequence[Doc]) -> str:
    blocks = []
    for doc in docs:
        lines = [f"[Doc {doc.id}] {doc.title}"]
        lines += [f"  [{i}] {sentence}" for i, sentence in enumerate(doc.sentences)]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_judge(claim: str, docs: Sequence[Doc], problem: str | None = None, version: str | None = None) -> str:
    feedback = FEEDBACK_TEMPLATE.format(problem=problem) if problem else ""
    template = JUDGE_TEMPLATES[version or JUDGE_VERSION]
    return template.format(claim=claim, abstracts=render_abstracts(docs), feedback=feedback)


def render_relation(claim: str, sentences: Sequence[str], version: str | None = None) -> str:
    template = RELATION_TEMPLATES[version or RELATION_VERSION]
    return template.format(claim=claim, evidence="\n".join(f"- {s}" for s in sentences))


def config_fingerprint(judge_version: str | None = None, relation_version: str | None = None, **settings) -> str:
    """Hash of the selected prompt texts plus every setting that shapes the run (k, thinking, policy, ...)."""
    judge_version, relation_version = judge_version or JUDGE_VERSION, relation_version or RELATION_VERSION
    payload = {
        "judge": [judge_version, JUDGE_TEMPLATES[judge_version]],
        "feedback": FEEDBACK_TEMPLATE,
        "relation": [relation_version, RELATION_TEMPLATES[relation_version]],
        "settings": settings,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
