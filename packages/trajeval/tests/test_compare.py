"""Tests for trajeval.compare."""

from __future__ import annotations

from datetime import UTC, datetime

from trajeval.compare import check_regressions, diff_aggregates, format_comparison_table
from trajeval.config import RegressionDirection, RegressionThreshold
from trajeval.results import RunMetadata, RunResult


def _run_result(aggregate_scores: dict) -> RunResult:
    metadata = RunMetadata(
        run_id="r",
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        config_hash="x",
        adapter_name="a",
        num_trajectories=0,
        num_errors=0,
        metric_names=list(aggregate_scores),
    )
    return RunResult(metadata=metadata, trajectory_results=[], aggregate_scores=aggregate_scores)


def test_diff_aggregates_computes_delta() -> None:
    baseline = _run_result({"retrieval_necessity": {"necessity_score": 0.8}})
    candidate = _run_result({"retrieval_necessity": {"necessity_score": 0.9}})
    deltas = diff_aggregates(baseline, candidate)
    assert len(deltas) == 1
    d = deltas[0]
    assert d.baseline_value == 0.8
    assert d.candidate_value == 0.9
    assert abs(d.delta - 0.1) < 1e-9


def test_diff_aggregates_degenerate_key_only_in_one_run() -> None:
    baseline = _run_result({"retrieval_necessity": {"necessity_score": 0.8}})
    candidate = _run_result({"retrieval_necessity": {"necessity_score": 0.8, "new_key": 5}})
    deltas = diff_aggregates(baseline, candidate)
    new_key_delta = next(d for d in deltas if d.key == "new_key")
    assert new_key_delta.baseline_value is None
    assert new_key_delta.candidate_value == 5
    assert new_key_delta.delta is None


def test_diff_aggregates_degenerate_non_numeric_value_ignored() -> None:
    baseline = _run_result({"recovery": {"outcome_distribution": "n/a"}})
    candidate = _run_result({"recovery": {"outcome_distribution": "n/a"}})
    deltas = diff_aggregates(baseline, candidate)
    assert deltas[0].baseline_value is None
    assert deltas[0].candidate_value is None


def test_diff_aggregates_degenerate_both_empty() -> None:
    assert diff_aggregates(_run_result({}), _run_result({})) == []


def test_check_regressions_flags_drop_beyond_tolerance_higher_is_better() -> None:
    baseline = _run_result({"retrieval_necessity": {"necessity_score": 0.9}})
    candidate = _run_result({"retrieval_necessity": {"necessity_score": 0.5}})
    deltas = diff_aggregates(baseline, candidate)
    thresholds = [
        RegressionThreshold(
            metric="retrieval_necessity",
            key="necessity_score",
            tolerance=0.05,
            direction=RegressionDirection.HIGHER_IS_BETTER,
        )
    ]
    checked = check_regressions(deltas, thresholds)
    assert checked[0].regressed is True


def test_check_regressions_within_tolerance_not_flagged() -> None:
    baseline = _run_result({"retrieval_necessity": {"necessity_score": 0.90}})
    candidate = _run_result({"retrieval_necessity": {"necessity_score": 0.87}})
    deltas = diff_aggregates(baseline, candidate)
    thresholds = [
        RegressionThreshold(metric="retrieval_necessity", key="necessity_score", tolerance=0.05)
    ]
    checked = check_regressions(deltas, thresholds)
    assert checked[0].regressed is False


def test_check_regressions_lower_is_better_flags_increase() -> None:
    baseline = _run_result({"termination": {"mean_excess_steps": 1.0}})
    candidate = _run_result({"termination": {"mean_excess_steps": 3.0}})
    deltas = diff_aggregates(baseline, candidate)
    thresholds = [
        RegressionThreshold(
            metric="termination",
            key="mean_excess_steps",
            tolerance=0.5,
            direction=RegressionDirection.LOWER_IS_BETTER,
        )
    ]
    checked = check_regressions(deltas, thresholds)
    assert checked[0].regressed is True


def test_check_regressions_no_threshold_configured_never_flags() -> None:
    baseline = _run_result({"retrieval_necessity": {"necessity_score": 0.9}})
    candidate = _run_result({"retrieval_necessity": {"necessity_score": 0.1}})
    deltas = diff_aggregates(baseline, candidate)
    checked = check_regressions(deltas, thresholds=[])
    assert all(not d.regressed for d in checked)


def test_check_regressions_degenerate_empty_deltas() -> None:
    assert check_regressions([], []) == []


def test_format_comparison_table_contains_rows_and_marker() -> None:
    baseline = _run_result({"retrieval_necessity": {"necessity_score": 0.9}})
    candidate = _run_result({"retrieval_necessity": {"necessity_score": 0.1}})
    deltas = diff_aggregates(baseline, candidate)
    checked = check_regressions(
        deltas, [RegressionThreshold(metric="retrieval_necessity", key="necessity_score")]
    )
    table = format_comparison_table(checked)
    assert "retrieval_necessity" in table
    assert "necessity_score" in table
    assert "regressed" in table


def test_format_comparison_table_degenerate_empty() -> None:
    table = format_comparison_table([])
    assert "Metric" in table  # header still renders
