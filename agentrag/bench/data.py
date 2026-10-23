"""Download and load a BEIR dataset (SciFact by default).

The archive comes from a third party, so it is extracted into its own fresh
directory after rejecting any member that would escape it.
"""
from __future__ import annotations

import csv
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DEFAULT_CACHE = Path(".bench_cache")


@dataclass
class BeirDataset:
    name: str
    corpus: dict[str, str]          # doc id -> "title text"
    queries: dict[str, str]         # query id -> text (test queries only)
    qrels: dict[str, dict[str, int]]  # query id -> {doc id -> relevance}


def _safe_extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    root = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if not (target / member).resolve().is_relative_to(root):
                raise RuntimeError(f"Refusing to extract unsafe path from archive: {member}")
        zf.extractall(target)


def download(name: str = "scifact", cache_dir: Path = DEFAULT_CACHE) -> Path:
    """Return the dataset directory, downloading and extracting it if needed."""
    dataset_dir = cache_dir / name / name
    if (dataset_dir / "corpus.jsonl").exists():
        return dataset_dir

    archive = cache_dir / f"{name}.zip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        try:
            # The BEIR host omits its intermediate certificate; the OS trust
            # store can fetch it, Python's bundled CA list cannot. Verification
            # stays on either way.
            import truststore

            truststore.inject_into_ssl()
        except ImportError:
            pass
        urllib.request.urlretrieve(BEIR_URL.format(name=name), archive)  # noqa: S310 - fixed https URL
    _safe_extract(archive, cache_dir / name)
    return dataset_dir


def load(name: str = "scifact", cache_dir: Path = DEFAULT_CACHE, split: str = "test") -> BeirDataset:
    root = download(name, cache_dir)

    corpus: dict[str, str] = {}
    with open(root / "corpus.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            corpus[row["_id"]] = f"{row.get('title', '')} {row.get('text', '')}".strip()

    qrels: dict[str, dict[str, int]] = {}
    with open(root / "qrels" / f"{split}.tsv", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # header: query-id, corpus-id, score
        for qid, doc_id, score in reader:
            qrels.setdefault(qid, {})[doc_id] = int(score)

    queries: dict[str, str] = {}
    with open(root / "queries.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["_id"] in qrels:
                queries[row["_id"]] = row["text"]

    return BeirDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)
