"""Tests for trajeval.metrics.termination."""

from __future__ import annotations

from trajeval.metrics.termination import TerminationMetric
from trajeval.types import (
    AnswerStep,
    GoldenRecord,
    RetrievalStep,
    RetrievedChunk,
    ThoughtStep,
    Trajectory,
)

metric = TerminationMetric()


def _golden(**kwargs) -> GoldenRecord:
    defaults = dict(
        id="g", question="q", reference_answer="a", retrieval_required=True, min_steps=2
    )
    defaults.update(kwargs)
    return GoldenRecord(**defaults)


def test_answers_immediately_after_sufficient_zero_excess() -> None:
    g = _golden(required_doc_ids=["doc-1"])
    t = Trajectory(
        question="q",
        final_answer="a",
        steps=[
            RetrievalStep(query="q", chunks=[RetrievedChunk(doc_id="doc-1", text="t")]),
            AnswerStep(text="a"),
        ],
    )
    result = metric.score(t, g)
    assert result.value == 0.0


def test_excess_steps_after_becoming_sufficient() -> None:
    g = _golden(required_doc_ids=["doc-1"])
    t = Trajectory(
        question="q",
        final_answer="a",
        steps=[
            RetrievalStep(query="q", chunks=[RetrievedChunk(doc_id="doc-1", text="t")]),
            ThoughtStep(text="let me double check"),
            RetrievalStep(query="q again", chunks=[]),
            AnswerStep(text="a"),
        ],
    )
    result = metric.score(t, g)
    assert result.value == 2.0  # steps at index 1 and 2, before answering at index 3
    assert result.details["sufficient_at_step"] == 0
    assert result.details["answered_at_step"] == 3


def test_or_semantics_sufficient_on_first_matching_alternative() -> None:
    g = _golden(sufficient_doc_ids=["doc-a", "doc-b"])
    t = Trajectory(
        question="q",
        final_answer="a",
        steps=[
            RetrievalStep(query="q", chunks=[RetrievedChunk(doc_id="doc-b", text="t")]),
            AnswerStep(text="a"),
        ],
    )
    result = metric.score(t, g)
    assert result.value == 0.0


def test_degenerate_retrieval_not_required() -> None:
    g = _golden(retrieval_required=False)
    t = Trajectory(question="q", final_answer="a", steps=[AnswerStep(text="a")])
    result = metric.score(t, g)
    assert result.value is None


def test_degenerate_never_becomes_sufficient() -> None:
    g = _golden(required_doc_ids=["doc-1"])
    t = Trajectory(
        question="q",
        final_answer="a",
        steps=[
            RetrievalStep(query="q", chunks=[RetrievedChunk(doc_id="wrong-doc", text="t")]),
            AnswerStep(text="a"),
        ],
    )
    result = metric.score(t, g)
    assert result.value is None


def test_degenerate_no_answer_step() -> None:
    g = _golden(required_doc_ids=["doc-1"])
    t = Trajectory(
        question="q",
        final_answer="a",
        steps=[RetrievalStep(query="q", chunks=[RetrievedChunk(doc_id="doc-1", text="t")])],
    )
    result = metric.score(t, g)
    assert result.value is None


def test_degenerate_empty_trajectory() -> None:
    g = _golden(required_doc_ids=["doc-1"])
    t = Trajectory(question="q", final_answer="a", steps=[])
    result = metric.score(t, g)
    assert result.value is None


def test_aggregate_degenerate_empty() -> None:
    agg = metric.aggregate([])
    assert agg["applicable"] == 0
    assert agg["mean_excess_steps"] is None


def test_aggregate_mean_excess_steps() -> None:
    g = _golden(required_doc_ids=["doc-1"])
    zero_excess = Trajectory(
        question="q",
        final_answer="a",
        steps=[
            RetrievalStep(query="q", chunks=[RetrievedChunk(doc_id="doc-1", text="t")]),
            AnswerStep(text="a"),
        ],
    )
    two_excess = Trajectory(
        question="q",
        final_answer="a",
        steps=[
            RetrievalStep(query="q", chunks=[RetrievedChunk(doc_id="doc-1", text="t")]),
            ThoughtStep(text="..."),
            ThoughtStep(text="..."),
            AnswerStep(text="a"),
        ],
    )
    results = [metric.score(zero_excess, g), metric.score(two_excess, g)]
    agg = metric.aggregate(results)
    assert agg["applicable"] == 2
    assert agg["mean_excess_steps"] == 1.0
