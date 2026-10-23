"""The original SciFact release: claims with SUPPORT / CONTRADICT evidence labels.

BEIR's SciFact "test" queries are the original release's *dev* claims. This
module loads the original data, which carries what BEIR drops (claim-level
labels and gold rationale sentences), and ``check_beir_parity`` proves the two
line up so retrieval numbers and verdict numbers describe the same 300 claims.

The archive is pinned by SHA-256 and extracted with path checks, since it is
third-party data.
"""
from __future__ import annotations

import hashlib
import json
import random
import tarfile
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SCIFACT_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
DEFAULT_CACHE = Path(".bench_cache")

SUPPORT, CONTRADICT, NEI = "SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"
LABELS = (SUPPORT, CONTRADICT, NEI)


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join((self.title, *self.sentences)).strip()


@dataclass
class Claim:
    id: str
    text: str
    label: str
    # doc id -> sentence indices a human marked as the rationale (empty for NOT_ENOUGH_INFO)
    rationale: dict[str, set[int]] = field(default_factory=dict)
    cited_doc_ids: tuple[str, ...] = ()

    @property
    def evidence_docs(self) -> set[str]:
        return set(self.rationale)


def claim_label(evidence: Mapping[str, Iterable[Mapping]]) -> str:
    """Claim-level label from per-document evidence; no evidence means NOT_ENOUGH_INFO."""
    labels = {item["label"] for items in evidence.values() for item in items}
    if not labels:
        return NEI
    if len(labels) > 1:
        raise ValueError(f"claim has mixed evidence labels: {sorted(labels)}")
    return labels.pop()


def parse_claim(row: Mapping) -> Claim:
    evidence = row.get("evidence") or {}
    rationale: dict[str, set[int]] = {}
    for doc_id, items in evidence.items():
        rationale[str(doc_id)] = {int(i) for item in items for i in item["sentences"]}
    return Claim(
        id=str(row["id"]),
        text=row["claim"].strip(),
        label=claim_label(evidence),
        rationale=rationale,
        cited_doc_ids=tuple(str(d) for d in row.get("cited_doc_ids", [])),
    )


@dataclass
class SciFact:
    docs: dict[str, Doc]
    train: list[Claim]
    dev: list[Claim]

    def label_counts(self, claims: Iterable[Claim]) -> dict[str, int]:
        counts = Counter(c.label for c in claims)
        return {label: counts.get(label, 0) for label in LABELS}


def load_from_dir(root: Path) -> SciFact:
    """Parse an extracted release directory (the one holding ``claims_dev.jsonl``)."""

    def read(name: str) -> list[dict]:
        with open(root / name, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    docs = {
        str(row["doc_id"]): Doc(str(row["doc_id"]), row["title"].strip(), tuple(s.strip() for s in row["abstract"]))
        for row in read("corpus.jsonl")
    }
    return SciFact(
        docs=docs,
        train=[parse_claim(r) for r in read("claims_train.jsonl")],
        dev=[parse_claim(r) for r in read("claims_dev.jsonl")],
    )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    root = target.resolve()
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            if not (target / member.name).resolve().is_relative_to(root) or member.issym() or member.islnk():
                raise RuntimeError(f"Refusing to extract unsafe archive member: {member.name}")
        tf.extractall(target)  # noqa: S202 - every member was checked above


def download(cache_dir: Path = DEFAULT_CACHE, expected_sha256: str = SCIFACT_SHA256) -> Path:
    """Return the extracted data directory, downloading and verifying the archive if needed."""
    base = cache_dir / "scifact_original"
    data_dir = base / "data"
    if (data_dir / "claims_dev.jsonl").exists():
        return data_dir

    archive = cache_dir / "scifact_original.tar.gz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        try:
            import truststore  # the host's chain is only complete in the OS trust store on some machines

            truststore.inject_into_ssl()
        except ImportError:
            pass
        urllib.request.urlretrieve(SCIFACT_URL, archive)  # noqa: S310 - fixed https URL

    actual = sha256_of(archive)
    if actual != expected_sha256:
        raise RuntimeError(
            f"SciFact archive checksum mismatch: expected {expected_sha256}, got {actual}. "
            f"Delete {archive} to re-download, or investigate if the release changed."
        )
    _safe_extract(archive, base)
    return data_dir


def load(cache_dir: Path = DEFAULT_CACHE) -> SciFact:
    return load_from_dir(download(cache_dir))


def check_beir_parity(sf: SciFact, beir_queries: Mapping[str, str], beir_corpus_ids: Iterable[str]) -> dict[str, int]:
    """Assert BEIR's test queries are exactly the original dev claims, with identical text and corpus.

    Raises ``AssertionError`` with the first differences if not, so a silent
    mismatch can never contaminate results.
    """
    dev = {c.id: c for c in sf.dev}
    missing = sorted(set(beir_queries) - set(dev))
    extra = sorted(set(dev) - set(beir_queries))
    assert not missing and not extra, f"id mismatch: only in BEIR {missing[:5]}, only in dev {extra[:5]}"
    differing = [qid for qid, text in beir_queries.items() if text.strip() != dev[qid].text]
    assert not differing, f"{len(differing)} claim texts differ, e.g. {differing[:5]}"
    corpus_a, corpus_b = set(sf.docs), set(beir_corpus_ids)
    assert corpus_a == corpus_b, f"corpus ids differ: {len(corpus_a ^ corpus_b)} documents"
    return {"claims": len(dev), "identical_text": len(dev), "documents": len(corpus_a)}


def stratified_subset(claims: Iterable[Claim], per_label: int, seed: int = 0) -> list[Claim]:
    """``per_label`` claims of each label, chosen with a seeded shuffle (for prompt development)."""
    rng = random.Random(seed)
    chosen: list[Claim] = []
    for label in LABELS:
        pool = sorted((c for c in claims if c.label == label), key=lambda c: int(c.id))
        rng.shuffle(pool)
        if len(pool) < per_label:
            raise ValueError(f"only {len(pool)} {label} claims, need {per_label}")
        chosen.extend(pool[:per_label])
    return sorted(chosen, key=lambda c: int(c.id))
