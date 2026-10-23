import json

import pytest

from agentrag.verify.cache import FIELDS, CachedResponse, CacheMiss, ResponseCache, find_secrets, make_key
from agentrag.verify.gemini import GeminiClient
from agentrag.verify.ratelimit import Limits, RateLimiter


def _key(**overrides):
    base = dict(model="m", prompt_version="v1", temperature=0.0, thinking="low", schema="JudgeOutput", prompt="p")
    base.update(overrides)
    return make_key(**base)


def test_key_is_stable_and_sensitive_to_everything_that_can_change_the_answer():
    assert _key() == _key()
    for change in (
        {"model": "m2"}, {"prompt_version": "v2"}, {"temperature": 0.5}, {"thinking": "medium"},
        {"schema": "RelationOutput"}, {"prompt": "p2"}, {"attempt": 1},
    ):
        assert _key(**change) != _key(), change


def test_put_get_and_persistence_across_instances(tmp_path):
    path = tmp_path / "cache.jsonl"
    cache = ResponseCache(path)
    entry = CachedResponse(key="k1", label="judge:v1", text='{"verdict":"SUPPORT"}', prompt_tokens=10, output_tokens=5)
    assert cache.get("k1") is None
    cache.put(entry)
    assert cache.get("k1") == entry and "k1" in cache and len(cache) == 1
    assert ResponseCache(path).get("k1") == entry


def test_a_partial_last_line_from_an_interrupted_write_is_skipped(tmp_path):
    path = tmp_path / "cache.jsonl"
    good = json.dumps({"key": "k1", "label": "x", "text": "ok"})
    path.write_text(good + '\n{"key": "k2", "label": "x", "te', encoding="utf-8")
    cache = ResponseCache(path)
    assert cache.get("k1").text == "ok" and cache.get("k2") is None


def test_only_declared_fields_are_ever_written(tmp_path):
    path = tmp_path / "cache.jsonl"
    ResponseCache(path).put(CachedResponse(key="k", label="l", text="t"))
    assert set(json.loads(path.read_text(encoding="utf-8"))) == set(FIELDS)


def test_offline_mode_serves_hits_and_refuses_misses(tmp_path):
    cache = ResponseCache(tmp_path / "c.jsonl")
    cache.put(CachedResponse(key="hit", label="l", text='{"relation": "neutral"}'))
    offline = ResponseCache(tmp_path / "c.jsonl", offline=True)
    assert offline.get("hit") is not None

    from agentrag.verify.schemas import RelationOutput

    client = GeminiClient("m", offline, RateLimiter(Limits(rpm=None, tpm=None)))
    with pytest.raises(CacheMiss):
        client.generate("never cached", RelationOutput, prompt_version="v1", label="x")


def test_find_secrets_flags_credentials_but_not_ordinary_text_and_never_echoes_them():
    planted = "Authorization: Bearer abcdefghijklmnopqrstuvwxyz and AIza" + "A" * 35
    hits = find_secrets(planted)
    assert hits and not any("abcdefghijklmnop" in h for h in hits)
    assert find_secrets("Venules have a thinner smooth muscle layer than arterioles.") == []
    assert find_secrets('{"verdict": "SUPPORT", "citations": [{"doc_id": "14717500", "sentence_ids": [2]}]}') == []
