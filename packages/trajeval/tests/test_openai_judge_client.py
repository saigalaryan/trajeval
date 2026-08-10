"""Tests for OpenAIJudgeClient's real call path — request construction,
caching, cost tracking, retry configuration — using a stand-in for
`openai.OpenAI` so no network call ever happens. Deliberately mirrors
test_anthropic_judge_client.py: same fixtures shape, same test names where
the behavior is identical, since both clients share a constructor contract
(see judge/client.py's module docstring) and should behave identically from
a caller's point of view.
"""

from __future__ import annotations

from pathlib import Path

from trajeval.cost import CostTracker
from trajeval.judge.cache import JudgeCache
from trajeval.judge.client import OpenAIJudgeClient


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeChatCompletion:
    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.choices = [_FakeChoice(text)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


class _FakeCompletionsResource:
    def __init__(self, response_text: str, prompt_tokens: int, completion_tokens: int) -> None:
        self._response_text = response_text
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self.calls: list[dict] = []

    def create(self, **kwargs):  # noqa: ANN003 - mirrors the real SDK's **kwargs signature
        self.calls.append(kwargs)
        return _FakeChatCompletion(
            self._response_text, self._prompt_tokens, self._completion_tokens
        )


class _FakeChatResource:
    def __init__(self, response_text: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.completions = _FakeCompletionsResource(response_text, prompt_tokens, completion_tokens)


class _FakeOpenAISDKClient:
    """Stand-in for `openai.OpenAI`."""

    def __init__(
        self,
        *,
        max_retries: int = 2,
        response_text: str = "ok",
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
    ) -> None:
        self.max_retries = max_retries
        self.chat = _FakeChatResource(response_text, prompt_tokens, completion_tokens)


def _install_fake_sdk(monkeypatch, **kwargs) -> None:
    def factory(*, max_retries: int = 2) -> _FakeOpenAISDKClient:
        return _FakeOpenAISDKClient(max_retries=max_retries, **kwargs)

    monkeypatch.setattr("openai.OpenAI", factory)


def test_judge_sends_prompt_and_returns_text(monkeypatch) -> None:
    _install_fake_sdk(monkeypatch, response_text="the answer")
    client = OpenAIJudgeClient(model="gpt-4o-mini")

    result = client.judge("what is 2+2?")

    assert result == "the answer"
    sent = client._client.chat.completions.calls[0]  # noqa: SLF001 - white-box test of this class's own wiring
    assert sent["model"] == "gpt-4o-mini"
    assert sent["messages"] == [{"role": "user", "content": "what is 2+2?"}]


def test_judge_includes_system_prompt_as_a_system_message(monkeypatch) -> None:
    _install_fake_sdk(monkeypatch, response_text="ok")
    client = OpenAIJudgeClient()

    client.judge("prompt", system="you are a judge")

    sent = client._client.chat.completions.calls[0]  # noqa: SLF001
    assert sent["messages"] == [
        {"role": "system", "content": "you are a judge"},
        {"role": "user", "content": "prompt"},
    ]


def test_judge_passes_max_retries_to_sdk_client(monkeypatch) -> None:
    _install_fake_sdk(monkeypatch)
    client = OpenAIJudgeClient(max_retries=7)
    assert client._client.max_retries == 7  # noqa: SLF001


def test_judge_uses_max_completion_tokens_configured(monkeypatch) -> None:
    _install_fake_sdk(monkeypatch)
    client = OpenAIJudgeClient(max_tokens=256)
    client.judge("prompt")
    assert client._client.chat.completions.calls[0]["max_completion_tokens"] == 256  # noqa: SLF001


def test_judge_degenerate_no_cache_calls_every_time(monkeypatch) -> None:
    _install_fake_sdk(monkeypatch, response_text="resp")
    client = OpenAIJudgeClient()

    client.judge("same prompt")
    client.judge("same prompt")

    assert len(client._client.chat.completions.calls) == 2  # noqa: SLF001


def test_judge_uses_cache_on_second_identical_call(monkeypatch, tmp_path: Path) -> None:
    _install_fake_sdk(monkeypatch, response_text="cached response")
    cache = JudgeCache(tmp_path / "cache.json")
    client = OpenAIJudgeClient(cache=cache)

    first = client.judge("same prompt")
    second = client.judge("same prompt")

    assert first == second == "cached response"
    assert len(client._client.chat.completions.calls) == 1  # noqa: SLF001 - only one real call made


def test_judge_cache_miss_on_different_prompt(monkeypatch, tmp_path: Path) -> None:
    _install_fake_sdk(monkeypatch, response_text="resp")
    cache = JudgeCache(tmp_path / "cache.json")
    client = OpenAIJudgeClient(cache=cache)

    client.judge("prompt a")
    client.judge("prompt b")

    assert len(client._client.chat.completions.calls) == 2  # noqa: SLF001


def test_judge_cache_persists_across_client_instances(monkeypatch, tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    _install_fake_sdk(monkeypatch, response_text="first response")
    client1 = OpenAIJudgeClient(cache=JudgeCache(cache_path))
    client1.judge("prompt")

    _install_fake_sdk(monkeypatch, response_text="should never be seen")
    client2 = OpenAIJudgeClient(cache=JudgeCache(cache_path))
    result = client2.judge("prompt")

    assert result == "first response"
    assert client2._client.chat.completions.calls == []  # noqa: SLF001 - no call made at all


def test_judge_records_cost_via_tracker(monkeypatch) -> None:
    _install_fake_sdk(monkeypatch, response_text="ok", prompt_tokens=123, completion_tokens=45)
    tracker = CostTracker()
    client = OpenAIJudgeClient(model="gpt-4o-mini", cost_tracker=tracker, label="recovery")

    client.judge("prompt")

    summary = tracker.summary()
    assert summary["recovery"]["calls"] == 1
    assert summary["recovery"]["input_tokens"] == 123
    assert summary["recovery"]["output_tokens"] == 45


def test_judge_cache_hit_does_not_record_cost(monkeypatch, tmp_path: Path) -> None:
    _install_fake_sdk(monkeypatch, response_text="cached", prompt_tokens=100, completion_tokens=50)
    cache = JudgeCache(tmp_path / "cache.json")
    tracker = CostTracker()
    client = OpenAIJudgeClient(cache=cache, cost_tracker=tracker, label="m")

    client.judge("prompt")
    client.judge("prompt")  # cache hit — must not double-record cost

    assert tracker.summary()["m"]["calls"] == 1


def test_judge_degenerate_no_cost_tracker_does_not_raise(monkeypatch) -> None:
    _install_fake_sdk(monkeypatch, response_text="ok")
    client = OpenAIJudgeClient()  # no cost_tracker passed
    assert client.judge("prompt") == "ok"


def test_judge_degenerate_empty_content_returns_empty_string(monkeypatch) -> None:
    """The SDK types `message.content` as possibly None (e.g. a
    finish_reason other than "stop") — must not crash trying to concatenate
    or index into it."""
    _install_fake_sdk(monkeypatch)
    client = OpenAIJudgeClient()

    def create(**kwargs):
        completion = _FakeChatCompletion("", 1, 1)
        completion.choices[0].message.content = None
        return completion

    client._client.chat.completions.create = create  # noqa: SLF001
    assert client.judge("prompt") == ""
