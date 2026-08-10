"""The LLM judge seam: every judged metric depends on `JudgeClient`, never on
the Anthropic SDK (or any provider's SDK) directly.

This mirrors `trajeval.adapters.AgentAdapter` — a one-method Protocol that
lets metrics be unit-tested against `FakeJudgeClient` with zero API calls and
zero cost, and lets a real deployment point at whatever provider it wants.
`AnthropicJudgeClient` and `OpenAIJudgeClient` are the two bundled real
implementations; both take the same constructor shape (`model`, `max_retries`,
`max_tokens`, `cache`, `cost_tracker`, `label`) so `trajeval.config`'s
`judge_factory` can swap between them without the rest of the codebase
knowing which one it got.

Retry: both real clients rely on their own SDK's built-in retry (exponential
backoff on 429/5xx/connection errors, configurable via `max_retries`) rather
than a hand-rolled retry loop — each SDK already does this correctly, so a
second implementation would just be a second place for it to be subtly wrong.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from trajeval.judge.cache import JudgeCache, cache_key


@runtime_checkable
class JudgeClient(Protocol):
    """Anything that can answer a single judge prompt with text."""

    def judge(self, prompt: str, *, system: str | None = None) -> str: ...


class JudgeParseError(ValueError):
    """Raised when a judge's response can't be parsed as the JSON it was
    asked to produce. Judged metrics should let this propagate rather than
    silently coercing a malformed response into a default score."""


def extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON value from a judge response.

    Judges are asked to respond with "only JSON" but reliably wrap it in
    markdown fences or add a sentence of preamble anyway. This strips
    fences and takes the first balanced ``{...}`` or ``[...]`` span rather
    than assuming the whole response is clean JSON.
    """
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    start_chars = "{["
    start = next((i for i, c in enumerate(stripped) if c in start_chars), None)
    if start is None:
        raise JudgeParseError(f"no JSON object/array found in judge response: {text!r}")
    end_char = "}" if stripped[start] == "{" else "]"
    end = stripped.rfind(end_char)
    if end == -1 or end < start:
        raise JudgeParseError(f"unbalanced JSON in judge response: {text!r}")
    candidate = stripped[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"invalid JSON in judge response: {candidate!r}") from exc


class AnthropicJudgeClient:
    """Judge backed by the real Anthropic API.

    Defaults to ``claude-opus-5``; pass ``model`` to use a different one
    (e.g. a cheaper model for high-volume judging). Requires the
    ``anthropic`` package and a resolvable API credential (env var or
    ``ant auth login`` profile) — see the ``claude-api`` skill for details.
    """

    def __init__(
        self,
        model: str = "claude-opus-5",
        *,
        max_retries: int = 5,
        max_tokens: int = 1024,
        cache: JudgeCache | None = None,
        cost_tracker: Any | None = None,
        label: str = "judge",
    ) -> None:
        import anthropic  # local import: don't require the SDK to import trajeval at all

        self._client = anthropic.Anthropic(max_retries=max_retries)
        self._model = model
        self._max_tokens = max_tokens
        self._cache = cache
        # Typed as Any to avoid a hard import of trajeval.cost.CostTracker
        # here — this module shouldn't need to know that type exists, only
        # that it's something with a .record(label, model, in, out) method.
        self._cost_tracker = cost_tracker
        self._label = label

    def judge(self, prompt: str, *, system: str | None = None) -> str:
        key = cache_key(prompt=prompt, system=system, model=self._model)
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return cached  # a cache hit made no API call — nothing to cost-track

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        text = "".join(block.text for block in response.content if block.type == "text")

        if self._cost_tracker is not None:
            self._cost_tracker.record(
                self._label, self._model, response.usage.input_tokens, response.usage.output_tokens
            )
        if self._cache is not None:
            self._cache.set(key, text)
        return text


class OpenAIJudgeClient:
    """Judge backed by the OpenAI API (Chat Completions).

    Defaults to ``gpt-4o-mini`` — a cheap, fast default; pass ``model`` for
    something else. Requires the ``openai`` package and a resolvable API
    credential (``OPENAI_API_KEY`` env var, or however the SDK finds one).

    Same constructor shape as `AnthropicJudgeClient` deliberately — see the
    module docstring — so switching `trajeval.yaml`'s ``judge_provider``
    doesn't require touching anything else.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        max_retries: int = 5,
        max_tokens: int = 1024,
        cache: JudgeCache | None = None,
        cost_tracker: Any | None = None,
        label: str = "judge",
    ) -> None:
        import openai  # local import: don't require the SDK to import trajeval at all

        self._client = openai.OpenAI(max_retries=max_retries)
        self._model = model
        self._max_tokens = max_tokens
        self._cache = cache
        # Typed as Any for the same reason as AnthropicJudgeClient's
        # cost_tracker — see there.
        self._cost_tracker = cost_tracker
        self._label = label

    def judge(self, prompt: str, *, system: str | None = None) -> str:
        key = cache_key(prompt=prompt, system=system, model=self._model)
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return cached  # a cache hit made no API call — nothing to cost-track

        messages: list[dict[str, Any]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # kwargs typed as dict[str, Any] and passed via **kwargs, same as
        # AnthropicJudgeClient — the SDK's `create()` overloads are typed
        # against a closed union of TypedDicts per role, which a plain
        # list[dict] will never satisfy under mypy --strict no matter how
        # it's constructed; Any is the honest escape hatch here, not a sign
        # something's wrong.
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_completion_tokens": self._max_tokens,
            "messages": messages,
        }
        response = self._client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""

        if self._cost_tracker is not None and response.usage is not None:
            self._cost_tracker.record(
                self._label,
                self._model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
        if self._cache is not None:
            self._cache.set(key, text)
        return text


class FakeJudgeClient:
    """Deterministic judge for tests: no network calls, no cost.

    `responses` maps an exact prompt string to its canned response, or is a
    callable taking the prompt and returning one. Use this to unit-test
    judged metrics' scoring logic without depending on real model output —
    what the *judge* says is trusted; what the *metric* does with it is
    what's under test.
    """

    def __init__(self, responses: dict[str, str] | Callable[[str], str]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, str | None]] = []

    def judge(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        if callable(self._responses):
            return self._responses(prompt)
        if prompt in self._responses:
            return self._responses[prompt]
        raise KeyError(
            f"FakeJudgeClient has no canned response for prompt: {prompt!r}\n"
            f"Known prompts: {list(self._responses.keys())!r}"
        )
