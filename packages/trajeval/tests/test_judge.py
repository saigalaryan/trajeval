"""Tests for trajeval.judge (cache, JSON extraction, FakeJudgeClient).

No network calls anywhere in this file — AnthropicJudgeClient itself isn't
exercised here since that would require real credentials; it's a thin
wrapper and its logic (caching, prompt construction) is covered indirectly
through the cache tests and the judged metrics' tests, which all use
FakeJudgeClient.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trajeval.judge.cache import JudgeCache, cache_key
from trajeval.judge.client import FakeJudgeClient, JudgeParseError, extract_json

# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


def test_extract_json_clean_object() -> None:
    assert extract_json('{"score": 5}') == {"score": 5}


def test_extract_json_with_markdown_fence() -> None:
    text = '```json\n{"score": 3, "reason": "ok"}\n```'
    assert extract_json(text) == {"score": 3, "reason": "ok"}


def test_extract_json_with_preamble_sentence() -> None:
    text = 'Sure, here is my evaluation:\n{"score": 4}'
    assert extract_json(text) == {"score": 4}


def test_extract_json_array() -> None:
    assert extract_json('["a", "b", "c"]') == ["a", "b", "c"]


def test_extract_json_degenerate_no_json_at_all() -> None:
    with pytest.raises(JudgeParseError):
        extract_json("I cannot evaluate this.")


def test_extract_json_degenerate_malformed_json() -> None:
    with pytest.raises(JudgeParseError):
        extract_json('{"score": 5,}')  # trailing comma, invalid JSON


# ---------------------------------------------------------------------------
# JudgeCache
# ---------------------------------------------------------------------------


def test_cache_key_is_deterministic() -> None:
    k1 = cache_key(prompt="hello", system="sys", model="m")
    k2 = cache_key(prompt="hello", system="sys", model="m")
    assert k1 == k2


def test_cache_key_differs_on_any_input_change() -> None:
    base = cache_key(prompt="hello", system="sys", model="m")
    assert base != cache_key(prompt="hello!", system="sys", model="m")
    assert base != cache_key(prompt="hello", system="other", model="m")
    assert base != cache_key(prompt="hello", system="sys", model="m2")


def test_cache_in_memory_roundtrip() -> None:
    cache = JudgeCache()
    assert cache.get("k") is None
    cache.set("k", "v")
    assert cache.get("k") == "v"
    assert len(cache) == 1


def test_cache_persists_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    cache = JudgeCache(path)
    cache.set("k", "v")

    reloaded = JudgeCache(path)
    assert reloaded.get("k") == "v"


def test_cache_degenerate_missing_file_starts_empty(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "does-not-exist-yet.json")
    assert cache.get("anything") is None
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# FakeJudgeClient
# ---------------------------------------------------------------------------


def test_fake_judge_client_dict_lookup() -> None:
    judge = FakeJudgeClient({"is this good?": '{"score": 5}'})
    assert judge.judge("is this good?") == '{"score": 5}'
    assert judge.calls == [("is this good?", None)]


def test_fake_judge_client_callable() -> None:
    judge = FakeJudgeClient(lambda prompt: f"echo: {prompt}")
    assert judge.judge("hi", system="sys") == "echo: hi"
    assert judge.calls == [("hi", "sys")]


def test_fake_judge_client_degenerate_unknown_prompt_raises() -> None:
    judge = FakeJudgeClient({"known": "response"})
    with pytest.raises(KeyError):
        judge.judge("unknown prompt")
