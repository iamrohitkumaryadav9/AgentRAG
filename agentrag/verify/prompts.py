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

# judge-v2 is the single revision allowed after judge-v1 scored macro-F1 0.853 on the 60 train claims with every one
# of its 9 errors a false NOT_ENOUGH_INFO (precision 1.00 on SUPPORT and CONTRADICT). It targets that bias with
# general guidance on how to decide, nothing specific to those claims. v1 above is unchanged, so cached v1 results
# stay valid.
JUDGE_TEMPLATES["judge-v2"] = """You are verifying a scientific claim against retrieved paper abstracts.

Choose exactly one verdict:
- SUPPORT: the abstracts report a finding that agrees with the claim.
- CONTRADICT: the abstracts report a finding that disagrees with the claim.
- NOT_ENOUGH_INFO: the abstracts report no finding about the relationship the claim asserts.

How to decide:
- Claims are usually short paraphrases or direct consequences of findings in a paper, rarely word for word. \
Look for a sentence that reports a result about the same entities and the same relationship (the same intervention, \
exposure or factor, and the same outcome).
- If you find one, choose SUPPORT or CONTRADICT by whether that result agrees or disagrees with the claim, even \
when the wording, the level of detail or the population differs somewhat, or when one short, obvious step connects \
them. For example, a finding that X reduces Y contradicts "X has no effect on Y", and a finding that X protects \
against Y supports "X prevents Y".
- Choose NOT_ENOUGH_INFO only when you cannot point to such a sentence: the abstracts discuss the entities \
separately, or a different outcome or intervention, or are silent on the relationship. Do not stretch: if linking the \
evidence to the claim would take several assumptions, answer NOT_ENOUGH_INFO.
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
Respond with JSON only."""

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
