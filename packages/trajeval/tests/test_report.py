"""Tests for trajeval.report."""

from __future__ import annotations

from datetime import UTC, datetime

from trajeval.adapters import CallableAdapter
from trajeval.metrics.base import MetricResult
from trajeval.metrics.retrieval_necessity import RetrievalNecessityMetric
from trajeval.report import render_report
from trajeval.results import CalibrationState, RunMetadata, RunResult, TrajectoryResult
from trajeval.runner import run
from trajeval.types import AnswerStep, GoldenRecord, RetrievalStep, Trajectory


def _metadata(**kwargs) -> RunMetadata:
    defaults = dict(
        run_id="r1",
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        config_hash="x",
        adapter_name="a",
        num_trajectories=1,
        num_errors=0,
        metric_names=["retrieval_necessity"],
    )
    defaults.update(kwargs)
    return RunMetadata(**defaults)


def test_render_report_degenerate_empty_run() -> None:
    result = RunResult(
        metadata=_metadata(num_trajectories=0), trajectory_results=[], aggregate_scores={}
    )
    html = render_report(result)
    assert "trajeval report" in html
    assert "No trajectories to show" in html


def test_render_report_escapes_untrusted_content() -> None:
    """A question/answer containing '<script>' must never appear unescaped
    — this is a report meant to be pasted into a PR and viewed in a browser."""
    goldens = [
        GoldenRecord(
            id="g1",
            question="<script>alert(1)</script>",
            reference_answer="a",
            retrieval_required=False,
            min_steps=1,
        )
    ]

    def adapter_fn(question: str) -> dict:
        return {
            "final_answer": "<img src=x onerror=alert(1)>",
            "steps": [{"step_type": "answer", "text": "<img src=x onerror=alert(1)>"}],
        }

    result = run(CallableAdapter(adapter_fn), goldens, [RetrievalNecessityMetric()])
    html = render_report(result)
    assert "<script>alert(1)</script>" not in html
    assert "<img src=x onerror=alert(1)>" not in html
    assert "&lt;script&gt;" in html


def test_render_report_shows_uncalibrated_badge_for_judged_metric() -> None:
    metadata = _metadata(metric_names=["recovery"])
    tr = TrajectoryResult(
        golden_id="g1",
        question="q",
        trajectory_id="t1",
        trajectory=Trajectory(
            question="q",
            final_answer="a",
            steps=[RetrievalStep(query="q", chunks=[]), AnswerStep(text="a")],
        ),
        metric_results={
            "recovery": MetricResult(
                metric_name="recovery", value=1.0, details={"outcome": "correctly_abstained"}
            )
        },
    )
    result = RunResult(
        metadata=metadata,
        trajectory_results=[tr],
        aggregate_scores={"recovery": {"applicable": 1, "answered_from_bad_context_rate": 0.0}},
        calibration={"recovery": CalibrationState(is_calibrated=False, n_labels=0)},
    )
    html = render_report(result)
    assert "uncalibrated" in html


def test_render_report_shows_calibrated_badge_with_kappa() -> None:
    metadata = _metadata(metric_names=["recovery"])
    result = RunResult(
        metadata=metadata,
        trajectory_results=[],
        aggregate_scores={"recovery": {"applicable": 0}},
        calibration={"recovery": CalibrationState(is_calibrated=True, kappa=0.83, n_labels=50)},
    )
    html = render_report(result)
    assert "κ=0.83" in html
    assert "n=50" in html


def test_render_report_includes_worst_trajectory_trace() -> None:
    goldens = [
        GoldenRecord(
            id="g1",
            question="What is 2+2?",
            reference_answer="4",
            retrieval_required=False,
            min_steps=1,
        )
    ]

    def adapter_fn(question: str) -> dict:
        return {"final_answer": "4", "steps": [{"step_type": "answer", "text": "4"}]}

    result = run(CallableAdapter(adapter_fn), goldens, [RetrievalNecessityMetric()])
    html = render_report(result)
    assert "What is 2+2?" in html
    assert "Final answer:" in html
    assert "4" in html


def test_render_report_errored_trajectory_excluded_from_worst_but_shown_separately() -> None:
    goldens = [
        GoldenRecord(
            id="g1", question="q", reference_answer="a", retrieval_required=False, min_steps=1
        )
    ]

    def failing_adapter(question: str) -> dict:
        raise RuntimeError("boom")

    result = run(CallableAdapter(failing_adapter), goldens, [RetrievalNecessityMetric()])
    html = render_report(result)
    # excluded from "worst" (nothing to be worst at — no score was ever produced)
    assert "No trajectories to show" in html
    # but not silently dropped from the report entirely
    assert "Errored trajectories (1)" in html
    assert "RuntimeError: boom" in html


def test_render_report_no_errors_section_when_run_is_clean() -> None:
    goldens = [
        GoldenRecord(
            id="g1", question="q", reference_answer="a", retrieval_required=False, min_steps=1
        )
    ]

    def adapter_fn(question: str) -> dict:
        return {"final_answer": "4", "steps": [{"step_type": "answer", "text": "4"}]}

    result = run(CallableAdapter(adapter_fn), goldens, [RetrievalNecessityMetric()])
    html = render_report(result)
    assert "Errored trajectories" not in html


def test_render_report_shows_metric_scoring_failure_on_an_otherwise_fine_trajectory() -> None:
    metadata = _metadata(metric_names=["recovery"])
    tr = TrajectoryResult(
        golden_id="g1",
        question="q",
        trajectory_id="t1",
        trajectory=Trajectory(
            question="q",
            final_answer="a",
            steps=[RetrievalStep(query="q", chunks=[]), AnswerStep(text="a")],
        ),
        metric_results={},
        metric_errors={"recovery": "JudgeParseError: no JSON object/array found"},
    )
    result = RunResult(
        metadata=metadata,
        trajectory_results=[tr],
        aggregate_scores={"recovery": {"applicable": 0}},
    )
    html = render_report(result)
    assert "recovery failed to score" in html
    assert "JudgeParseError" in html


def test_render_report_per_tag_breakdown_reflects_actual_tags() -> None:
    goldens = [
        GoldenRecord(
            id="g1",
            question="q1",
            reference_answer="a",
            retrieval_required=False,
            min_steps=1,
            tags=["easy"],
        ),
        GoldenRecord(
            id="g2",
            question="q2",
            reference_answer="a",
            retrieval_required=False,
            min_steps=1,
            tags=["hard"],
        ),
    ]

    def adapter_fn(question: str) -> dict:
        return {"final_answer": "4", "steps": [{"step_type": "answer", "text": "4"}]}

    result = run(CallableAdapter(adapter_fn), goldens, [RetrievalNecessityMetric()])
    html = render_report(result)
    assert "easy" in html
    assert "hard" in html


def test_render_report_cost_summary_absent_by_default() -> None:
    result = RunResult(
        metadata=_metadata(num_trajectories=0), trajectory_results=[], aggregate_scores={}
    )
    html = render_report(result)
    assert "No judge cost data" in html


def test_render_report_cost_summary_rendered_when_provided() -> None:
    result = RunResult(
        metadata=_metadata(num_trajectories=0), trajectory_results=[], aggregate_scores={}
    )
    cost_summary = {
        "recovery": {"calls": 3, "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.012}
    }
    html = render_report(result, cost_summary=cost_summary)
    assert "recovery" in html
    assert "0.012" in html


def test_render_report_tag_breakdown_covers_a_judged_metric() -> None:
    """_metric_instance's judged branch (constructing a throwaway metric
    with a FakeJudgeClient purely for its .aggregate()) only runs when a
    judged metric name shows up in a tagged breakdown."""
    metadata = _metadata(metric_names=["recovery"])
    tr = TrajectoryResult(
        golden_id="g1",
        question="q",
        tags=["easy"],
        trajectory_id="t1",
        trajectory=Trajectory(
            question="q",
            final_answer="a",
            steps=[RetrievalStep(query="q", chunks=[]), AnswerStep(text="a")],
        ),
        metric_results={
            "recovery": MetricResult(
                metric_name="recovery", value=1.0, details={"outcome": "correctly_abstained"}
            )
        },
    )
    result = RunResult(
        metadata=metadata,
        trajectory_results=[tr],
        aggregate_scores={"recovery": {"applicable": 1}},
    )
    html = render_report(result)
    assert "easy" in html
    assert "recovery" in html


def test_render_report_renders_tool_and_thought_steps() -> None:
    goldens = [
        GoldenRecord(
            id="g1", question="q", reference_answer="a", retrieval_required=False, min_steps=1
        )
    ]

    def adapter_fn(question: str) -> dict:
        return {
            "final_answer": "42",
            "steps": [
                {"step_type": "thought", "text": "let me compute this"},
                {
                    "step_type": "tool",
                    "tool_name": "calculator",
                    "args": {"expr": "21*2"},
                    "result": 42,
                },
                {"step_type": "answer", "text": "42"},
            ],
        }

    result = run(CallableAdapter(adapter_fn), goldens, [RetrievalNecessityMetric()])
    html = render_report(result)
    assert "thought: let me compute this" in html
    assert "calculator" in html
    assert "&rarr;" in html or "→" in html
