"""trajeval compare: diff two `RunResult`s' aggregate scores, and flag
regressions against configured thresholds.

Split from the CLI command so the diff/regression logic is testable without
going through Typer or touching disk — `trajeval.cli` and the GitHub Action
both just call `format_comparison_table` on the output of `check_regressions`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from trajeval.config import RegressionDirection, RegressionThreshold
from trajeval.results import RunResult


@dataclass(frozen=True)
class MetricDelta:
    metric_name: str
    key: str
    baseline_value: float | None
    candidate_value: float | None
    delta: float | None
    regressed: bool = False


def diff_aggregates(baseline: RunResult, candidate: RunResult) -> list[MetricDelta]:
    """Every (metric, key) pair present in either run's aggregate scores.
    A key present in only one run gets None on the other side rather than
    being dropped — a metric that disappeared between runs is itself worth
    seeing in the diff, not silently absent."""
    deltas: list[MetricDelta] = []
    metric_names = sorted(set(baseline.aggregate_scores) | set(candidate.aggregate_scores))
    for metric_name in metric_names:
        base_agg = baseline.aggregate_scores.get(metric_name, {})
        cand_agg = candidate.aggregate_scores.get(metric_name, {})
        keys = sorted(set(base_agg) | set(cand_agg))
        for key in keys:
            base_value = base_agg.get(key)
            cand_value = cand_agg.get(key)
            b = base_value if isinstance(base_value, int | float) else None
            c = cand_value if isinstance(cand_value, int | float) else None
            delta = (c - b) if (b is not None and c is not None) else None
            deltas.append(MetricDelta(metric_name, key, b, c, delta))
    return deltas


def check_regressions(
    deltas: list[MetricDelta], thresholds: list[RegressionThreshold]
) -> list[MetricDelta]:
    """Return a new list with `.regressed` set wherever a configured
    threshold is violated. A (metric, key) pair with no configured
    threshold is never flagged — regression checking is opt-in per key, not
    "anything that went down"."""
    by_key = {(t.metric, t.key): t for t in thresholds}
    result: list[MetricDelta] = []
    for d in deltas:
        threshold = by_key.get((d.metric_name, d.key))
        regressed = False
        if threshold is not None and d.delta is not None:
            if threshold.direction == RegressionDirection.HIGHER_IS_BETTER:
                regressed = d.delta < -threshold.tolerance
            else:
                regressed = d.delta > threshold.tolerance
        result.append(replace(d, regressed=regressed))
    return result


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def format_comparison_table(deltas: list[MetricDelta]) -> str:
    """Markdown table — what the GitHub Action posts as a PR comment."""
    lines = ["| Metric | Key | Baseline | Candidate | Delta | |", "|---|---|---|---|---|---|"]
    for d in deltas:
        marker = "🔴 regressed" if d.regressed else ""
        lines.append(
            f"| {d.metric_name} | {d.key} | {_fmt(d.baseline_value)} | "
            f"{_fmt(d.candidate_value)} | {_fmt(d.delta)} | {marker} |"
        )
    return "\n".join(lines)
