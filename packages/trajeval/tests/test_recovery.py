"""Tests for trajeval.metrics.recovery."""

from __future__ import annotations

from trajeval.judge.client import FakeJudgeClient
from trajeval.metrics.recovery import RecoveryMetric, RecoveryOutcome
from trajeval.types import AnswerStep, GoldenRecord, RetrievalStep, RetrievedChunk, Trajectory


def _golden(**kwargs) -> GoldenRecord:
    defaults = dict(
        id="g",
        question="What year was the bridge built?",
        reference_answer="1932",
        retrieval_required=True,
        min_steps=2,
        required_doc_ids=["doc-bridge"],
    )
    defaults.update(kwargs)
    return GoldenRecord(**defaults)


def test_reformulation_is_detected_deterministically_no_judge_call() -> None:
    judge = FakeJudgeClient({})  # should never be called
    metric = RecoveryMetric(judge)
    t = Trajectory(
        question="q",
        final_answer="1932",
        steps=[
            RetrievalStep(
                query="bridge history", chunks=[RetrievedChunk(doc_id="irrelevant", text="x")]
            ),
            RetrievalStep(
                query="completely different construction year query",
                chunks=[RetrievedChunk(doc_id="doc-bridge", text="1932")],
            ),
            AnswerStep(text="1932"),
        ],
    )
    result = metric.score(t, _golden())
    assert result.value == 1.0
    assert result.details["outcome"] == RecoveryOutcome.REFORMULATED.value
    assert judge.calls == []


def test_answered_from_bad_context_via_judge() -> None:
    judge = FakeJudgeClient({})

    def respond(prompt: str) -> str:
        return '{"outcome": "answered_from_bad_context", "reason": "test"}'

    judge = FakeJudgeClient(respond)
    metric = RecoveryMetric(judge)
    t = Trajectory(
        question="q",
        final_answer="The bridge was built in 1850, according to the search results.",
        steps=[
            RetrievalStep(
                query="bridge history",
                chunks=[RetrievedChunk(doc_id="irrelevant", text="unrelated content")],
            ),
            AnswerStep(text="The bridge was built in 1850, according to the search results."),
        ],
    )
    result = metric.score(t, _golden())
    assert result.value == 0.0
    assert result.details["outcome"] == RecoveryOutcome.ANSWERED_FROM_BAD_CONTEXT.value


def test_correctly_abstained_via_judge() -> None:
    def respond(prompt: str) -> str:
        return '{"outcome": "correctly_abstained", "reason": "test"}'

    judge = FakeJudgeClient(respond)
    metric = RecoveryMetric(judge)
    t = Trajectory(
        question="q",
        final_answer="I couldn't find reliable information about this.",
        steps=[
            RetrievalStep(
                query="bridge history",
                chunks=[RetrievedChunk(doc_id="irrelevant", text="unrelated content")],
            ),
            AnswerStep(text="I couldn't find reliable information about this."),
        ],
    )
    result = metric.score(t, _golden())
    assert result.value == 1.0
    assert result.details["outcome"] == RecoveryOutcome.CORRECTLY_ABSTAINED.value


def test_degenerate_no_bad_retrieval_not_applicable() -> None:
    judge = FakeJudgeClient({})
    metric = RecoveryMetric(judge)
    t = Trajectory(
        question="q",
        final_answer="1932",
        steps=[
            RetrievalStep(
                query="bridge history", chunks=[RetrievedChunk(doc_id="doc-bridge", text="1932")]
            ),
            AnswerStep(text="1932"),
        ],
    )
    result = metric.score(t, _golden())
    assert result.value is None
    assert judge.calls == []


def test_degenerate_no_retrieval_at_all_not_applicable() -> None:
    judge = FakeJudgeClient({})
    metric = RecoveryMetric(judge)
    t = Trajectory(question="q", final_answer="a", steps=[AnswerStep(text="a")])
    result = metric.score(t, _golden())
    assert result.value is None


def test_degenerate_retrieval_not_required_never_bad() -> None:
    """When retrieval_required=False there's no 'relevant doc' concept, so
    a retrieval step (over-retrieval) can never be classified 'bad' here —
    that failure mode belongs to retrieval_necessity, not recovery."""
    judge = FakeJudgeClient({})
    metric = RecoveryMetric(judge)
    g = _golden(retrieval_required=False, required_doc_ids=[])
    t = Trajectory(
        question="q",
        final_answer="a",
        steps=[
            RetrievalStep(query="q", chunks=[RetrievedChunk(doc_id="whatever", text="x")]),
            AnswerStep(text="a"),
        ],
    )
    result = metric.score(t, g)
    assert result.value is None


def test_aggregate_degenerate_empty() -> None:
    metric = RecoveryMetric(FakeJudgeClient({}))
    agg = metric.aggregate([])
    assert agg["applicable"] == 0
    assert agg["answered_from_bad_context_rate"] is None


def test_aggregate_reports_distribution() -> None:
    reformulated_judge = FakeJudgeClient({})
    metric = RecoveryMetric(reformulated_judge)
    reformulated_traj = Trajectory(
        question="q",
        final_answer="1932",
        steps=[
            RetrievalStep(
                query="bridge history", chunks=[RetrievedChunk(doc_id="irrelevant", text="x")]
            ),
            RetrievalStep(
                query="entirely different phrasing here",
                chunks=[RetrievedChunk(doc_id="doc-bridge", text="1932")],
            ),
            AnswerStep(text="1932"),
        ],
    )
    result1 = metric.score(reformulated_traj, _golden())

    def bad_respond(prompt: str) -> str:
        return '{"outcome": "answered_from_bad_context", "reason": "test"}'

    bad_metric = RecoveryMetric(FakeJudgeClient(bad_respond))
    bad_traj = Trajectory(
        question="q",
        final_answer="1850",
        steps=[
            RetrievalStep(
                query="bridge history", chunks=[RetrievedChunk(doc_id="irrelevant", text="x")]
            ),
            AnswerStep(text="1850"),
        ],
    )
    result2 = bad_metric.score(bad_traj, _golden())

    agg = metric.aggregate([result1, result2])
    assert agg["applicable"] == 2
    assert agg["reformulated"] == 1
    assert agg["answered_from_bad_context"] == 1
    assert agg["answered_from_bad_context_rate"] == 0.5
