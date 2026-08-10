"""Tests for trajeval.metrics.query_quality, using FakeJudgeClient so no
real model calls happen — the judge's opinion is trusted; what the metric
does with it is what's under test.
"""

from __future__ import annotations

from trajeval.judge.client import FakeJudgeClient
from trajeval.metrics.query_quality import QueryQualityMetric
from trajeval.types import AnswerStep, GoldenRecord, RetrievalStep, RetrievedChunk, Trajectory


def _golden(**kwargs) -> GoldenRecord:
    defaults = dict(
        id="g",
        question="q",
        reference_answer="a",
        retrieval_required=True,
        min_steps=1,
        required_doc_ids=["doc-1"],
    )
    defaults.update(kwargs)
    return GoldenRecord(**defaults)


def _judge_scoring(score_by_query: dict[str, int]) -> FakeJudgeClient:
    def respond(prompt: str) -> str:
        for query, score in score_by_query.items():
            if f'Query: "{query}"' in prompt:
                return f'{{"score": {score}, "reason": "test"}}'
        raise AssertionError(f"unexpected prompt: {prompt}")

    return FakeJudgeClient(respond)


def test_high_quality_query_scores_near_one() -> None:
    judge = _judge_scoring({"Eiffel Tower completion year": 5})
    metric = QueryQualityMetric(judge)
    t = Trajectory(
        question="q",
        final_answer="a",
        steps=[
            RetrievalStep(
                query="Eiffel Tower completion year",
                chunks=[RetrievedChunk(doc_id="doc-1", text="1889")],
            ),
            AnswerStep(text="1889"),
        ],
    )
    result = metric.score(t, _golden())
    assert result.value == 1.0  # (5-1)/4
    assert result.details["queries"][0]["hit"] is True


def test_low_quality_query_scores_near_zero() -> None:
    judge = _judge_scoring({"um can you find that thing": 1})
    metric = QueryQualityMetric(judge)
    t = Trajectory(
        question="q",
        final_answer="a",
        steps=[RetrievalStep(query="um can you find that thing", chunks=[]), AnswerStep(text="a")],
    )
    result = metric.score(t, _golden())
    assert result.value == 0.0  # (1-1)/4
    assert result.details["queries"][0]["hit"] is False


def test_deterministic_hit_reported_separately_from_judged_quality() -> None:
    """A well-formed query that happens to miss the relevant doc: hit=False
    but judged_quality can still be high — the two must never be conflated."""
    judge = _judge_scoring({"good query wrong target": 5})
    metric = QueryQualityMetric(judge)
    t = Trajectory(
        question="q",
        final_answer="a",
        steps=[
            RetrievalStep(
                query="good query wrong target",
                chunks=[RetrievedChunk(doc_id="unrelated-doc", text="x")],
            )
        ],
    )
    result = metric.score(t, _golden())
    assert result.details["queries"][0]["hit"] is False
    assert result.details["queries"][0]["judged_quality"] == 5


def test_degenerate_no_retrieval_steps() -> None:
    judge = FakeJudgeClient({})
    metric = QueryQualityMetric(judge)
    t = Trajectory(question="q", final_answer="a", steps=[AnswerStep(text="a")])
    result = metric.score(t, _golden())
    assert result.value is None
    assert judge.calls == []


def test_multiple_queries_averaged() -> None:
    judge = _judge_scoring({"query one": 5, "query two": 1})
    metric = QueryQualityMetric(judge)
    t = Trajectory(
        question="q",
        final_answer="a",
        steps=[
            RetrievalStep(query="query one", chunks=[]),
            RetrievalStep(query="query two", chunks=[]),
        ],
    )
    result = metric.score(t, _golden())
    assert result.value == 0.5  # mean of (5-1)/4=1.0 and (1-1)/4=0.0


def test_aggregate_degenerate_empty() -> None:
    metric = QueryQualityMetric(FakeJudgeClient({}))
    agg = metric.aggregate([])
    assert agg["applicable"] == 0
    assert agg["mean_judged_quality"] is None
    assert agg["hit_rate"] is None


def test_aggregate_hit_rate_and_mean_quality() -> None:
    judge = _judge_scoring({"q1": 5, "q2": 1})
    metric = QueryQualityMetric(judge)
    t1 = Trajectory(
        question="q",
        final_answer="a",
        steps=[RetrievalStep(query="q1", chunks=[RetrievedChunk(doc_id="doc-1", text="x")])],
    )
    t2 = Trajectory(question="q", final_answer="a", steps=[RetrievalStep(query="q2", chunks=[])])
    results = [metric.score(t1, _golden()), metric.score(t2, _golden())]
    agg = metric.aggregate(results)
    assert agg["applicable"] == 2
    assert agg["hit_rate"] == 0.5
    assert agg["mean_judged_quality"] == 0.5
