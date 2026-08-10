"""Tests for AnthropicJudgeClient's real call path — request construction,
caching, cost tracking, retry configuration — using a stand-in for
`anthropic.Anthropic` so no network call ever happens. `FakeJudgeClient`
(used everywhere else) covers the `JudgeClient` *contract*; this file is
specifically about `AnthropicJudgeClient`'s own wiring to the SDK.
"""

from __future__ import annotations

from pathlib import Path

from trajeval.cost import CostTracker
from trajeval.judge.cache import JudgeCache
from trajeval.judge.client import AnthropicJudgeClient


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeMessage:
    def __init__(self, text: str, input_tokens: int, output_tokens: int) -> None:
        self.content = [_FakeTextBlock(text)]
        self.usage = _FakeUsage(input_tokens, output_tokens)


class _FakeMessagesResource:
    def __init__(self, response_text: str, input_tokens: int, output_tokens: int) -> None:
        self._response_text = response_text
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.calls: list[dict] = []

    def create(self, **kwargs):  # noqa: ANN003 - mirrors the real SDK's **kwargs signature
        self.calls.append(kwargs)
        return _FakeMessage(self._response_text, self._input_tokens, self._output_tokens)


class _FakeAnthropicSDKClient:
    """Stand-in for `anthropic.Anthropic`."""

    def __init__(
        self,
        *,
        max_retries: int = 2,
        response_text: str = "ok",
        input_tokens: int = 10,
        output_tokens: int = 5,
    ) -> None:
        self.max_retries = max_retries
        self.messages = _FakeMessagesResource(response_text, input_tokens, output_tokens)


def _install_fake_sdk(monkeypatch, **kwargs) -> None:
    def factory(*, max_retries: int = 2) -> _FakeAnthropicSDKClient:
        return _FakeAnthropicSDKClient(max_retries=max_retries, **kwargs)

    monkeypatch.setattr("anthropic.Anthropic", factory)


def test_judge_sends_prompt_and_returns_text(monkeypatch) -> None:
    _install_fake_sdk(monkeypatch, response_text="the answer")
    client = AnthropicJudgeClient(model="claude-haiku-4-5")

    result = client.judge("what is 2+2?")

    assert result == "the answer"
    sent = client._client.messages.calls[0]  # noqa: SLF001 - white-box test of this class's own wiring
    assert sent["model"] == "claude-haiku-4-5"
    assert sent["messages"] == [{"role": "user", "content": "what is 2+2?"}]
    assert "system" not in sent


def test_judge_includes_system_prompt_when_given(monkeypatch) -> None:
    _install_fake_sdk(monkeypatch, response_text="ok")
    client = AnthropicJudgeClient()

    client.judge("prompt", system="you are a judge")

    sent = client._client.messages.calls[0]  # noqa: SLF001
    assert sent["system"] == "you are a judge"


def test_judge_passes_max_retries_to_sdk_client(monkeypatch) -> None:
    _install_fake_sdk(monkeypatch)
    client = AnthropicJudgeClient(max_retries=7)
    assert client._client.max_retries == 7  # noqa: SLF001


def test_judge_uses_max_tokens_configured(monkeypatch) -> None:
    _install_fake_sdk(monkeypatch)
    client = AnthropicJudgeClient(max_tokens=256)
    client.judge("prompt")
    assert client._client.messages.calls[0]["max_tokens"] == 256  # noqa: SLF001


def test_judge_degenerate_no_cache_calls_every_time(monkeypatch) -> None:
    _install_fake_sdk(monkeypatch, response_text="resp")
    client = AnthropicJudgeClient()

    client.judge("same prompt")
    client.judge("same prompt")

    assert len(client._client.messages.calls) == 2  # noqa: SLF001


def test_judge_uses_cache_on_second_identical_call(monkeypatch, tmp_path: Path) -> None:
    _install_fake_sdk(monkeypatch, response_text="cached response")
    cache = JudgeCache(tmp_path / "cache.json")
    client = AnthropicJudgeClient(cache=cache)

    first = client.judge("same prompt")
    second = client.judge("same prompt")

    assert first == second == "cached response"
    assert len(client._client.messages.calls) == 1  # noqa: SLF001 - only one real call made


def test_judge_cache_miss_on_different_prompt(monkeypatch, tmp_path: Path) -> None:
    _install_fake_sdk(monkeypatch, response_text="resp")
    cache = JudgeCache(tmp_path / "cache.json")
    client = AnthropicJudgeClient(cache=cache)

    client.judge("prompt a")
    client.judge("prompt b")

    assert len(client._client.messages.calls) == 2  # noqa: SLF001


def test_judge_cache_persists_across_client_instances(monkeypatch, tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    _install_fake_sdk(monkeypatch, response_text="first response")
    client1 = AnthropicJudgeClient(cache=JudgeCache(cache_path))
    client1.judge("prompt")

    # A second client, same cache file, backed by an SDK that would return
    # something different if actually called — the cache hit must win.
    _install_fake_sdk(monkeypatch, response_text="should never be seen")
    client2 = AnthropicJudgeClient(cache=JudgeCache(cache_path))
    result = client2.judge("prompt")

    assert result == "first response"
    assert client2._client.messages.calls == []  # noqa: SLF001 - no call made at all


def test_judge_records_cost_via_tracker(monkeypatch) -> None:
    _install_fake_sdk(monkeypatch, response_text="ok", input_tokens=123, output_tokens=45)
    tracker = CostTracker()
    client = AnthropicJudgeClient(model="claude-opus-5", cost_tracker=tracker, label="recovery")

    client.judge("prompt")

    summary = tracker.summary()
    assert summary["recovery"]["calls"] == 1
    assert summary["recovery"]["input_tokens"] == 123
    assert summary["recovery"]["output_tokens"] == 45


def test_judge_cache_hit_does_not_record_cost(monkeypatch, tmp_path: Path) -> None:
    _install_fake_sdk(monkeypatch, response_text="cached", input_tokens=100, output_tokens=50)
    cache = JudgeCache(tmp_path / "cache.json")
    tracker = CostTracker()
    client = AnthropicJudgeClient(cache=cache, cost_tracker=tracker, label="m")

    client.judge("prompt")
    client.judge("prompt")  # cache hit — must not double-record cost

    assert tracker.summary()["m"]["calls"] == 1


def test_judge_degenerate_no_cost_tracker_does_not_raise(monkeypatch) -> None:
    _install_fake_sdk(monkeypatch, response_text="ok")
    client = AnthropicJudgeClient()  # no cost_tracker passed
    assert client.judge("prompt") == "ok"


def test_judge_concatenates_multiple_text_blocks(monkeypatch) -> None:
    """Defensive: a response split across multiple text blocks (rare but
    valid per the SDK's content-block model) must be joined, not truncated
    to the first block."""
    _install_fake_sdk(monkeypatch)
    client = AnthropicJudgeClient()

    def create(**kwargs):
        msg = _FakeMessage("", 1, 1)
        msg.content = [_FakeTextBlock("hello "), _FakeTextBlock("world")]
        return msg

    client._client.messages.create = create  # noqa: SLF001
    assert client.judge("prompt") == "hello world"
