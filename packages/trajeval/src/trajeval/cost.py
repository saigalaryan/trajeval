"""Per-run token and cost accounting for judge calls, broken down by metric.

Deterministic metrics make no LLM calls, so there's nothing to account for
on their side. Judged metrics report usage through `AnthropicJudgeClient`'s
`cost_tracker=`/`label=` constructor args — the natural pattern is one
`AnthropicJudgeClient` instance per judged metric, so `label` is typically
just the metric's name and the resulting breakdown lines up with
`RunResult.aggregate_scores`.

Pricing is a local table, not fetched live — treat `cost_usd` as an
estimate, and expect to update the table when Anthropic's or OpenAI's
pricing changes. An unrecognized model records tokens correctly but flags
cost as unknown rather than silently reporting $0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import JsonValue

# USD per 1,000,000 tokens: (input_price, output_price).
PRICING_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # OpenAI, for OpenAIJudgeClient. Verified against
    # developers.openai.com/api/docs/pricing (standard tier, short
    # context) — re-check there if these ever look stale; OpenAI, like
    # Anthropic, doesn't publish pricing anywhere this table can fetch
    # live from.
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """None means "model not in the pricing table", not "free"."""
    pricing = PRICING_PER_MILLION_TOKENS.get(model)
    if pricing is None:
        return None
    input_price, output_price = pricing
    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price


@dataclass
class _Bucket:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cost_unknown: bool = False


@dataclass
class CostTracker:
    """Accumulates token usage and estimated cost per label."""

    _buckets: dict[str, _Bucket] = field(default_factory=dict)

    def record(self, label: str, model: str, input_tokens: int, output_tokens: int) -> None:
        bucket = self._buckets.setdefault(label, _Bucket())
        bucket.calls += 1
        bucket.input_tokens += input_tokens
        bucket.output_tokens += output_tokens
        cost = estimate_cost_usd(model, input_tokens, output_tokens)
        if cost is None:
            bucket.cost_unknown = True
        else:
            bucket.cost_usd += cost

    def summary(self) -> dict[str, dict[str, JsonValue]]:
        return {
            label: {
                "calls": b.calls,
                "input_tokens": b.input_tokens,
                "output_tokens": b.output_tokens,
                "cost_usd": None if b.cost_unknown else round(b.cost_usd, 4),
            }
            for label, b in self._buckets.items()
        }

    def total_cost_usd(self) -> float | None:
        if not self._buckets:
            return 0.0
        if any(b.cost_unknown for b in self._buckets.values()):
            return None
        return round(sum(b.cost_usd for b in self._buckets.values()), 4)
