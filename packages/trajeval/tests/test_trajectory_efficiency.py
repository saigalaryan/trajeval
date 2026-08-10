"""Tests for trajeval.metrics.trajectory_efficiency."""

from __future__ import annotations

from trajeval.metrics.trajectory_efficiency import TrajectoryEfficiencyMetric, detect_loops
from trajeval.types import (
    AnswerStep,
    GoldenRecord,
    RetrievalStep,
    ThoughtStep,
    ToolStep,
    Trajectory,
)

metric = TrajectoryEfficiencyMetric()


def _golden(min_steps: int) -> GoldenRecord:
    return GoldenRecord(
        id="g", question="q", reference_answer="a", retrieval_required=True, min_steps=min_steps
    )


def test_optimal_trajectory_scores_one() -> None:
    t = Trajectory(
        question="q",
        final_answer="a",
        steps=[RetrievalStep(query="q"), AnswerStep(text="a")],
    )
    result = metric.score(t, _golden(min_steps=2))
    assert result.value == 1.0


def test_longer_than_optimal_scores_less_than_one() -> None:
    t = Trajectory(
        question="q",
        final_answer="a",
        steps=[
            ThoughtStep(text="hmm"),
            RetrievalStep(query="q1"),
            RetrievalStep(query="totally different topic entirely"),
            AnswerStep(text="a"),
        ],
    )
    result = metric.score(t, _golden(min_steps=2))
    assert result.value == 0.5


def test_fewer_than_min_steps_clips_to_one_not_above() -> None:
    """An implausible trajectory shorter than min_steps still caps at 1.0 —
    min_steps is a floor, not a target to beat."""
    t = Trajectory(question="q", final_answer="a", steps=[AnswerStep(text="a")])
    result = metric.score(t, _golden(min_steps=5))
    assert result.value == 1.0


def test_degenerate_empty_trajectory_has_no_value() -> None:
    t = Trajectory(question="q", final_answer="a", steps=[])
    result = metric.score(t, _golden(min_steps=1))
    assert result.value is None


def test_degenerate_single_step() -> None:
    t = Trajectory(question="q", final_answer="a", steps=[AnswerStep(text="a")])
    result = metric.score(t, _golden(min_steps=1))
    assert result.value == 1.0


# ---------------------------------------------------------------------------
# loop detection
# ---------------------------------------------------------------------------


def test_detects_loop_of_near_identical_queries() -> None:
    t = Trajectory(
        question="q",
        final_answer="a",
        steps=[
            RetrievalStep(query="capital of france"),
            RetrievalStep(query="capital of france "),  # near-identical
            AnswerStep(text="a"),
        ],
    )
    loops = detect_loops(t)
    assert loops == [(0, 1)]


def test_thought_step_between_does_not_break_loop_detection() -> None:
    t = Trajectory(
        question="q",
        final_answer="a",
        steps=[
            RetrievalStep(query="capital of france"),
            ThoughtStep(text="let me think about this again"),
            RetrievalStep(query="capital of france"),
            AnswerStep(text="a"),
        ],
    )
    loops = detect_loops(t)
    assert loops == [(0, 2)]


def test_tool_step_between_breaks_loop_detection() -> None:
    """A tool call in between means the agent did something productive —
    that's not looping even if it searches the same thing again after."""
    t = Trajectory(
        question="q",
        final_answer="a",
        steps=[
            RetrievalStep(query="capital of france"),
            ToolStep(tool_name="calculator", args={}, result=None),
            RetrievalStep(query="capital of france"),
            AnswerStep(text="a"),
        ],
    )
    loops = detect_loops(t)
    assert loops == []


def test_no_loop_for_genuinely_different_queries() -> None:
    t = Trajectory(
        question="q",
        final_answer="a",
        steps=[
            RetrievalStep(query="capital of france"),
            RetrievalStep(query="population of tokyo"),
        ],
    )
    assert detect_loops(t) == []


def test_degenerate_no_retrieval_steps_no_loops() -> None:
    t = Trajectory(question="q", final_answer="a", steps=[AnswerStep(text="a")])
    assert detect_loops(t) == []


def test_metric_details_report_loop_presence() -> None:
    t = Trajectory(
        question="q",
        final_answer="a",
        steps=[RetrievalStep(query="x"), RetrievalStep(query="x"), AnswerStep(text="a")],
    )
    result = metric.score(t, _golden(min_steps=2))
    assert result.details["has_loop"] is True
    assert result.details["loop_count"] == 1


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def test_aggregate_degenerate_empty() -> None:
    agg = metric.aggregate([])
    assert agg["applicable"] == 0
    assert agg["mean_efficiency"] is None
    assert agg["loop_rate"] is None


def test_aggregate_computes_mean_and_loop_rate() -> None:
    optimal = Trajectory(question="q", final_answer="a", steps=[AnswerStep(text="a")])
    looping = Trajectory(
        question="q",
        final_answer="a",
        steps=[RetrievalStep(query="x"), RetrievalStep(query="x"), AnswerStep(text="a")],
    )
    g1 = _golden(min_steps=1)
    g2 = _golden(min_steps=2)
    results = [metric.score(optimal, g1), metric.score(looping, g2)]
    agg = metric.aggregate(results)
    assert agg["applicable"] == 2
    assert agg["loop_rate"] == 0.5
    assert agg["mean_efficiency"] == (1.0 + (2 / 3)) / 2
