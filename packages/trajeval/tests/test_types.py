"""Tests for trajeval.types.

Scope, deliberately: does the schema round-trip losslessly, and does it
enforce the invariants we just agreed on (mutually-exclusive doc-id sets,
retrieval_required consistency)? Metric behaviour is tested where the metrics
live, not here.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trajeval.types import (
    AnswerStep,
    GoldenRecord,
    RetrievalStep,
    RetrievedChunk,
    ThoughtStep,
    ToolStep,
    Trajectory,
    TrajectoryMetadata,
)

# ---------------------------------------------------------------------------
# Trajectory round-trip
# ---------------------------------------------------------------------------


def _full_trajectory() -> Trajectory:
    """A trajectory using every step type, for round-trip coverage."""
    return Trajectory(
        golden_id="g-1",
        question="What year was the Eiffel Tower completed?",
        final_answer="1889.",
        steps=[
            ThoughtStep(text="I don't know this off-hand, let me search."),
            RetrievalStep(
                query="Eiffel Tower completion year",
                chunks=[
                    RetrievedChunk(doc_id="doc-42", text="Completed in 1889.", score=0.91, rank=1),
                    RetrievedChunk(doc_id="doc-07", text="Located in Paris.", score=0.55, rank=2),
                ],
            ),
            ToolStep(
                tool_name="calculator",
                args={"expr": "1889 - 1887"},
                result={"value": 2},
            ),
            AnswerStep(text="1889."),
        ],
        metadata=TrajectoryMetadata(
            model="gpt-4o",
            prompt_tokens=120,
            completion_tokens=8,
            latency_ms=842.5,
            extra_provider_field="whatever",  # allowed via extra="allow"
        ),
    )


def test_trajectory_round_trips_losslessly() -> None:
    original = _full_trajectory()
    restored = Trajectory.model_validate_json(original.model_dump_json())
    assert restored == original


def test_trajectory_degenerate_empty_steps() -> None:
    """No retrieval, no steps at all — just a question and an answer."""
    t = Trajectory(question="What is 2 + 2?", final_answer="4")
    restored = Trajectory.model_validate_json(t.model_dump_json())
    assert restored == t
    assert restored.steps == []


def test_trajectory_single_step() -> None:
    t = Trajectory(
        question="What is 2 + 2?",
        final_answer="4",
        steps=[AnswerStep(text="4")],
    )
    restored = Trajectory.model_validate_json(t.model_dump_json())
    assert len(restored.steps) == 1
    assert isinstance(restored.steps[0], AnswerStep)


def test_trajectory_auto_generates_id_when_omitted() -> None:
    t1 = Trajectory(question="q", final_answer="a")
    t2 = Trajectory(question="q", final_answer="a")
    assert t1.id != t2.id  # each gets its own uuid4 by default


def test_step_discriminator_routes_to_correct_subclass() -> None:
    t = _full_trajectory()
    dumped = t.model_dump()
    restored = Trajectory.model_validate(dumped)
    assert [type(s) for s in restored.steps] == [ThoughtStep, RetrievalStep, ToolStep, AnswerStep]


def test_unknown_field_is_rejected() -> None:
    """extra='forbid' should catch typos/version drift, not silently drop them."""
    with pytest.raises(ValidationError):
        Trajectory.model_validate({"question": "q", "final_answer": "a", "not_a_real_field": True})


# ---------------------------------------------------------------------------
# GoldenRecord
# ---------------------------------------------------------------------------


def test_golden_record_round_trips() -> None:
    g = GoldenRecord(
        id="g-1",
        question="What year was the Eiffel Tower completed?",
        reference_answer="1889",
        required_doc_ids=["doc-42", "doc-07"],
        retrieval_required=True,
        min_steps=2,
        tags=["history", "easy"],
    )
    restored = GoldenRecord.model_validate_json(g.model_dump_json())
    assert restored == g


def test_golden_record_retrieval_not_required_degenerate_case() -> None:
    """The retrieval_required=False case this whole metric exists to catch."""
    g = GoldenRecord(
        id="g-2",
        question="What is the capital of France?",
        reference_answer="Paris",
        retrieval_required=False,
        min_steps=1,
        tags=["parametric"],
    )
    assert g.required_doc_ids == []
    assert g.sufficient_doc_ids == []


def test_required_and_sufficient_doc_ids_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError):
        GoldenRecord(
            id="g-3",
            question="q",
            reference_answer="a",
            required_doc_ids=["doc-1"],
            sufficient_doc_ids=["doc-2"],
            retrieval_required=True,
            min_steps=1,
        )


def test_retrieval_not_required_forbids_doc_ids() -> None:
    with pytest.raises(ValidationError):
        GoldenRecord(
            id="g-4",
            question="q",
            reference_answer="a",
            required_doc_ids=["doc-1"],
            retrieval_required=False,
            min_steps=1,
        )


def test_sufficient_doc_ids_alone_is_valid() -> None:
    g = GoldenRecord(
        id="g-5",
        question="q",
        reference_answer="a",
        sufficient_doc_ids=["doc-1", "doc-2"],
        retrieval_required=True,
        min_steps=1,
    )
    assert g.sufficient_doc_ids == ["doc-1", "doc-2"]
