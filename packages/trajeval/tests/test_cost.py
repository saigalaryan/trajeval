"""Tests for trajeval.cost."""

from __future__ import annotations

from trajeval.cost import CostTracker, estimate_cost_usd


def test_estimate_cost_known_model() -> None:
    # claude-opus-5: $5/$25 per 1M tokens
    cost = estimate_cost_usd("claude-opus-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 30.0


def test_estimate_cost_unknown_model_returns_none() -> None:
    assert estimate_cost_usd("some-future-model", 1000, 1000) is None


def test_estimate_cost_known_openai_model() -> None:
    # gpt-4o-mini: $0.15/$0.60 per 1M tokens
    cost = estimate_cost_usd("gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 0.75


def test_estimate_cost_degenerate_zero_tokens() -> None:
    assert estimate_cost_usd("claude-opus-5", 0, 0) == 0.0


def test_cost_tracker_records_and_summarizes() -> None:
    tracker = CostTracker()
    tracker.record("recovery", "claude-opus-5", 1000, 500)
    tracker.record("recovery", "claude-opus-5", 2000, 1000)

    summary = tracker.summary()
    assert summary["recovery"]["calls"] == 2
    assert summary["recovery"]["input_tokens"] == 3000
    assert summary["recovery"]["output_tokens"] == 1500
    assert summary["recovery"]["cost_usd"] == round((3000 / 1e6) * 5 + (1500 / 1e6) * 25, 4)


def test_cost_tracker_separates_labels() -> None:
    tracker = CostTracker()
    tracker.record("query_quality", "claude-haiku-4-5", 100, 50)
    tracker.record("faithfulness", "claude-opus-5", 100, 50)

    summary = tracker.summary()
    assert set(summary.keys()) == {"query_quality", "faithfulness"}
    assert summary["query_quality"]["calls"] == 1


def test_cost_tracker_degenerate_empty() -> None:
    tracker = CostTracker()
    assert tracker.summary() == {}
    assert tracker.total_cost_usd() == 0.0


def test_cost_tracker_unknown_model_flags_cost_none_not_zero() -> None:
    tracker = CostTracker()
    tracker.record("query_quality", "totally-unknown-model", 100, 50)
    summary = tracker.summary()
    assert summary["query_quality"]["cost_usd"] is None
    assert tracker.total_cost_usd() is None


def test_cost_tracker_total_cost_sums_all_labels() -> None:
    tracker = CostTracker()
    tracker.record("a", "claude-opus-5", 1_000_000, 0)  # $5
    tracker.record("b", "claude-haiku-4-5", 1_000_000, 0)  # $1
    assert tracker.total_cost_usd() == 6.0
