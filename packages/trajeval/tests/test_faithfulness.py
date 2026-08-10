"""Tests for trajeval.metrics.faithfulness."""

from __future__ import annotations

import json

from trajeval.judge.client import FakeJudgeClient
from trajeval.metrics.faithfulness import FaithfulnessMetric
from trajeval.types import AnswerStep, GoldenRecord, RetrievalStep, RetrievedChunk, Trajectory


def _golden() -> GoldenRecord:
    return GoldenRecord(
        id="g", question="q", reference_answer="a", retrieval_required=True, min_steps=1
    )


def _trajectory_with_chunks(answer: str) -> Trajectory:
    return Trajectory(
        question="q",
        final_answer=answer,
        steps=[
            RetrievalStep(
                query="q",
                chunks=[
                    RetrievedChunk(doc_id="doc-1", text="The Eiffel Tower was completed in 1889.")
                ],
            ),
            AnswerStep(text=answer),
        ],
    )


def test_all_claims_supported() -> None:
    def respond(prompt: str) -> str:
        if "Break the following answer" in prompt:
            return json.dumps(["The Eiffel Tower was completed in 1889."])
        return json.dumps(
            [
                {
                    "claim": "The Eiffel Tower was completed in 1889.",
                    "supported": True,
                    "reason": "matches context",
                }
            ]
        )

    metric = FaithfulnessMetric(FakeJudgeClient(respond))
    t = _trajectory_with_chunks("The Eiffel Tower was completed in 1889.")
    result = metric.score(t, _golden())
    assert result.value == 1.0
    assert len(result.details["claims"]) == 1


def test_some_claims_unsupported() -> None:
    def respond(prompt: str) -> str:
        if "Break the following answer" in prompt:
            return json.dumps(["The Eiffel Tower was completed in 1889.", "It is made of gold."])
        return json.dumps(
            [
                {
                    "claim": "The Eiffel Tower was completed in 1889.",
                    "supported": True,
                    "reason": "matches",
                },
                {"claim": "It is made of gold.", "supported": False, "reason": "not in context"},
            ]
        )

    metric = FaithfulnessMetric(FakeJudgeClient(respond))
    t = _trajectory_with_chunks("The Eiffel Tower was completed in 1889. It is made of gold.")
    result = metric.score(t, _golden())
    assert result.value == 0.5


def test_degenerate_no_retrieved_context() -> None:
    metric = FaithfulnessMetric(FakeJudgeClient({}))
    t = Trajectory(question="q", final_answer="4", steps=[AnswerStep(text="4")])
    result = metric.score(t, _golden())
    assert result.value is None


def test_degenerate_answer_decomposes_to_no_claims() -> None:
    def respond(prompt: str) -> str:
        assert "Break the following answer" in prompt
        return "[]"

    metric = FaithfulnessMetric(FakeJudgeClient(respond))
    t = _trajectory_with_chunks("I don't know.")
    result = metric.score(t, _golden())
    assert result.value is None
    assert result.details["claims"] == []


def test_aggregate_degenerate_empty() -> None:
    metric = FaithfulnessMetric(FakeJudgeClient({}))
    agg = metric.aggregate([])
    assert agg["applicable"] == 0
    assert agg["mean_faithfulness"] is None


def test_aggregate_counts_unsupported_claims_across_trajectories() -> None:
    def respond(prompt: str) -> str:
        if "Break the following answer" in prompt:
            return json.dumps(["claim a", "claim b"])
        return json.dumps(
            [
                {"claim": "claim a", "supported": True, "reason": "x"},
                {"claim": "claim b", "supported": False, "reason": "x"},
            ]
        )

    metric = FaithfulnessMetric(FakeJudgeClient(respond))
    t1 = _trajectory_with_chunks("claim a claim b")
    t2 = _trajectory_with_chunks("claim a claim b")
    results = [metric.score(t1, _golden()), metric.score(t2, _golden())]
    agg = metric.aggregate(results)
    assert agg["applicable"] == 2
    assert agg["total_claims"] == 4
    assert agg["unsupported_claims"] == 2
    assert agg["mean_faithfulness"] == 0.5
