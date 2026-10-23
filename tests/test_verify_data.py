import json
import tarfile

import pytest

from agentrag.verify import data
from agentrag.verify.data import (
    CONTRADICT,
    NEI,
    SUPPORT,
    check_beir_parity,
    claim_label,
    load_from_dir,
    parse_claim,
    sha256_of,
    stratified_subset,
)


def _write_release(root, dev_claims=None):
    root.mkdir(parents=True, exist_ok=True)
    corpus = [
        {"doc_id": 1, "title": "Title one", "abstract": ["First sentence.", "Second sentence."], "structured": False},
        {"doc_id": 2, "title": "Title two", "abstract": ["Only sentence."], "structured": False},
    ]
    train = [{"id": 10, "claim": "Train claim.", "evidence": {}, "cited_doc_ids": [1]}]
    dev = dev_claims or [
        {"id": 1, "claim": "Supported claim.", "cited_doc_ids": [1],
         "evidence": {"1": [{"sentences": [0, 1], "label": "SUPPORT"}, {"sentences": [1], "label": "SUPPORT"}]}},
        {"id": 3, "claim": "Contradicted claim.", "cited_doc_ids": [2],
         "evidence": {"2": [{"sentences": [0], "label": "CONTRADICT"}]}},
        {"id": 5, "claim": "No evidence claim.", "cited_doc_ids": [1], "evidence": {}},
    ]
    for name, rows in (("corpus", corpus), ("claims_train", train), ("claims_dev", dev)):
        (root / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return root


def test_claim_label_derives_from_evidence():
    assert claim_label({}) == NEI
    assert claim_label({"1": [{"sentences": [0], "label": "SUPPORT"}]}) == SUPPORT
    with pytest.raises(ValueError):
        claim_label({"1": [{"sentences": [0], "label": "SUPPORT"}], "2": [{"sentences": [1], "label": "CONTRADICT"}]})


def test_parse_claim_merges_rationale_sentences_per_document():
    evidence = {"7": [{"sentences": [2, 5], "label": "SUPPORT"}, {"sentences": [7], "label": "SUPPORT"}]}
    claim = parse_claim({"id": 1, "claim": " A claim. ", "cited_doc_ids": [7], "evidence": evidence})
    assert claim.text == "A claim."
    assert claim.rationale == {"7": {2, 5, 7}}
    assert claim.cited_doc_ids == ("7",)
    assert claim.evidence_docs == {"7"}


def test_load_from_dir_parses_docs_and_claims(tmp_path):
    sf = load_from_dir(_write_release(tmp_path / "data"))
    assert sf.docs["1"].sentences == ("First sentence.", "Second sentence.")
    assert sf.docs["1"].text == "Title one First sentence. Second sentence."
    assert [c.label for c in sf.dev] == [SUPPORT, CONTRADICT, NEI]
    assert sf.label_counts(sf.dev) == {SUPPORT: 1, CONTRADICT: 1, NEI: 1}
    assert len(sf.train) == 1


def test_beir_parity_passes_when_ids_text_and_corpus_match(tmp_path):
    sf = load_from_dir(_write_release(tmp_path / "data"))
    queries = {c.id: c.text for c in sf.dev}
    assert check_beir_parity(sf, queries, ["1", "2"]) == {"claims": 3, "identical_text": 3, "documents": 2}


def test_beir_parity_fails_loudly_on_any_difference(tmp_path):
    sf = load_from_dir(_write_release(tmp_path / "data"))
    queries = {c.id: c.text for c in sf.dev}
    with pytest.raises(AssertionError, match="id mismatch"):
        check_beir_parity(sf, {**queries, "99": "extra"}, ["1", "2"])
    with pytest.raises(AssertionError, match="texts differ"):
        check_beir_parity(sf, {**queries, "1": "changed"}, ["1", "2"])
    with pytest.raises(AssertionError, match="corpus ids differ"):
        check_beir_parity(sf, queries, ["1", "2", "3"])


def test_stratified_subset_is_balanced_seeded_and_errors_when_short(tmp_path):
    rows = [{"id": i, "claim": f"c{i}", "cited_doc_ids": [1], "evidence": {}} for i in range(40, 50)]
    rows += [{"id": i, "claim": f"s{i}", "cited_doc_ids": [1],
              "evidence": {"1": [{"sentences": [0], "label": "SUPPORT"}]}} for i in range(60, 70)]
    rows += [{"id": i, "claim": f"x{i}", "cited_doc_ids": [1],
              "evidence": {"1": [{"sentences": [0], "label": "CONTRADICT"}]}} for i in range(80, 90)]
    sf = load_from_dir(_write_release(tmp_path / "data", dev_claims=rows))
    first = stratified_subset(sf.dev, per_label=4, seed=1)
    assert [c.label for c in first].count(NEI) == 4
    assert first == stratified_subset(sf.dev, per_label=4, seed=1)
    assert first != stratified_subset(sf.dev, per_label=4, seed=2)
    with pytest.raises(ValueError):
        stratified_subset(sf.dev, per_label=11)


def test_download_rejects_a_checksum_mismatch_and_extracts_a_valid_archive(tmp_path):
    release = _write_release(tmp_path / "src" / "data")
    archive = tmp_path / "cache" / "scifact_original.tar.gz"
    archive.parent.mkdir()
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(release, arcname="data")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        data.download(tmp_path / "cache", expected_sha256="0" * 64)
    assert not (tmp_path / "cache" / "scifact_original").exists()  # nothing extracted on mismatch

    extracted = data.download(tmp_path / "cache", expected_sha256=sha256_of(archive))
    assert (extracted / "claims_dev.jsonl").exists()


def test_extraction_refuses_path_traversal(tmp_path):
    evil = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_text("x")
    with tarfile.open(evil, "w:gz") as tf:
        tf.add(payload, arcname="../escaped.txt")
    with pytest.raises(RuntimeError, match="unsafe"):
        data._safe_extract(evil, tmp_path / "out")
