"""Tests for trajeval.metrics.retrieval_necessity.

Every classification cell in the 2x2 table is covered, plus the degenerate
trajectories called out in the brief: empty (no steps at all) and
single-step. Aggregate is tested separately from classification so a bug in
one doesn't mask a bug in the other.
"""

from __future__ import annotations

from trajeval.metrics.retrieval_necessity import (
    RetrievalNecessityMetric,
    RetrievalOutcome,
    classify,
)
from trajeval.types import AnswerStep, GoldenRecord, RetrievalStep, ThoughtStep, Trajectory

metric = RetrievalNecessityMetric()


def _golden(*, retrieval_required: bool) -> GoldenRecord:
    return GoldenRecord(
        id="g-1",
        question="q",
        reference_answer="a",
        retrieval_required=retrieval_required,
        min_steps=1,
    )


def _trajectory(*, retrieved: bool) -> Trajectory:
    steps: list = [ThoughtStep(text="thinking")]
    if retrieved:
        steps.append(RetrievalStep(query="q"))
    steps.append(AnswerStep(text="a"))
    return Trajectory(question="q", final_answer="a", steps=steps)


# ---------------------------------------------------------------------------
# classify — the four cells
# ---------------------------------------------------------------------------


def test_correct_search() -> None:
    outcome = classify(_trajectory(retrieved=True), _golden(retrieval_required=True))
    assert outcome == RetrievalOutcome.CORRECT_SEARCH


def test_correct_skip() -> None:
    outcome = classify(_trajectory(retrieved=False), _golden(retrieval_required=False))
    assert outcome == RetrievalOutcome.CORRECT_SKIP


def test_over_retrieval() -> None:
    """Agent searched, but the question was answerable from parametric knowledge."""
    outcome = classify(_trajectory(retrieved=True), _golden(retrieval_required=False))
    assert outcome == RetrievalOutcome.OVER_RETRIEVAL


def test_under_retrieval() -> None:
    """Agent didn't search, but the question required it — likely hallucination."""
    outcome = classify(_trajectory(retrieved=False), _golden(retrieval_required=True))
    assert outcome == RetrievalOutcome.UNDER_RETRIEVAL


# ---------------------------------------------------------------------------
# classify — degenerate trajectories
# ---------------------------------------------------------------------------


def test_empty_trajectory_no_steps_at_all_counts_as_no_retrieval() -> None:
    t = Trajectory(question="q", final_answer="a", steps=[])
    assert classify(t, _golden(retrieval_required=False)) == RetrievalOutcome.CORRECT_SKIP
    assert classify(t, _golden(retrieval_required=True)) == RetrievalOutcome.UNDER_RETRIEVAL


def test_single_step_trajectory_answer_only() -> None:
    t = Trajectory(question="q", final_answer="a", steps=[AnswerStep(text="a")])
    assert classify(t, _golden(retrieval_required=False)) == RetrievalOutcome.CORRECT_SKIP


def test_single_step_trajectory_retrieval_only() -> None:
    t = Trajectory(question="q", final_answer="a", steps=[RetrievalStep(query="q")])
    assert classify(t, _golden(retrieval_required=True)) == RetrievalOutcome.CORRECT_SEARCH


def test_retrieval_step_with_zero_chunks_still_counts_as_retrieved() -> None:
    """A search that came back empty is still a search — that's a different
    failure (bad retrieval) from not searching at all (under-retrieval)."""
    t = Trajectory(question="q", final_answer="a", steps=[RetrievalStep(query="q", chunks=[])])
    assert classify(t, _golden(retrieval_required=True)) == RetrievalOutcome.CORRECT_SEARCH


# ---------------------------------------------------------------------------
# MetricResult.value
# ---------------------------------------------------------------------------


def test_score_value_is_one_for_correct_outcomes() -> None:
    result = metric.score(_trajectory(retrieved=True), _golden(retrieval_required=True))
    assert result.value == 1.0
    assert result.details["outcome"] == "correct_search"


def test_score_value_is_zero_for_failure_outcomes() -> None:
    result = metric.score(_trajectory(retrieved=True), _golden(retrieval_required=False))
    assert result.value == 0.0
    assert result.details["outcome"] == "over_retrieval"


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def test_aggregate_degenerate_empty_results() -> None:
    agg = metric.aggregate([])
    assert agg["total"] == 0
    assert agg["necessity_score"] is None
    assert agg["over_retrieval_rate"] is None
    assert agg["under_retrieval_rate"] is None


def test_aggregate_all_correct_gives_necessity_score_one() -> None:
    results = [
        metric.score(_trajectory(retrieved=True), _golden(retrieval_required=True)),
        metric.score(_trajectory(retrieved=False), _golden(retrieval_required=False)),
    ]
    agg = metric.aggregate(results)
    assert agg["necessity_score"] == 1.0
    assert agg["over_retrieval_rate"] == 0.0
    assert agg["under_retrieval_rate"] == 0.0


def test_aggregate_reports_over_and_under_rates_separately() -> None:
    """4 trajectories: 1 correct, 1 over-retrieval, 2 under-retrieval.

    Averaging over/under into one 'wrong' rate would say 75% failure either
    way, but the two failure modes have very different costs — this is the
    whole reason they're reported as separate rates.
    """
    results = [
        metric.score(_trajectory(retrieved=True), _golden(retrieval_required=True)),  # correct
        metric.score(_trajectory(retrieved=True), _golden(retrieval_required=False)),  # over
        metric.score(_trajectory(retrieved=False), _golden(retrieval_required=True)),  # under
        metric.score(_trajectory(retrieved=False), _golden(retrieval_required=True)),  # under
    ]
    agg = metric.aggregate(results)
    assert agg["total"] == 4
    assert agg["correct_search"] == 1
    assert agg["correct_skip"] == 0
    assert agg["over_retrieval"] == 1
    assert agg["under_retrieval"] == 2
    assert agg["necessity_score"] == 0.25
    assert agg["over_retrieval_rate"] == 0.25
    assert agg["under_retrieval_rate"] == 0.5
