"""Tests for trajeval.metrics.context."""

from __future__ import annotations

from trajeval.metrics.context import is_retrieval_adequate, relevant_doc_ids, step_chunk_ids
from trajeval.types import GoldenRecord, RetrievalStep, RetrievedChunk


def _golden(**kwargs) -> GoldenRecord:
    defaults = dict(
        id="g", question="q", reference_answer="a", retrieval_required=True, min_steps=1
    )
    defaults.update(kwargs)
    return GoldenRecord(**defaults)


def test_relevant_doc_ids_union_of_required_and_sufficient() -> None:
    # mutually exclusive in practice, but the helper just unions whatever's set
    g = _golden(required_doc_ids=["a", "b"])
    assert relevant_doc_ids(g) == {"a", "b"}


def test_step_chunk_ids() -> None:
    step = RetrievalStep(
        query="q",
        chunks=[RetrievedChunk(doc_id="x", text="t"), RetrievedChunk(doc_id="y", text="t")],
    )
    assert step_chunk_ids(step) == {"x", "y"}


def test_and_semantics_requires_all() -> None:
    g = _golden(required_doc_ids=["a", "b"])
    assert not is_retrieval_adequate({"a"}, g)
    assert is_retrieval_adequate({"a", "b"}, g)
    assert is_retrieval_adequate({"a", "b", "c"}, g)


def test_or_semantics_requires_any_one() -> None:
    g = _golden(sufficient_doc_ids=["a", "b"])
    assert is_retrieval_adequate({"a"}, g)
    assert is_retrieval_adequate({"b"}, g)
    assert not is_retrieval_adequate({"c"}, g)


def test_neither_set_is_trivially_adequate() -> None:
    g = _golden(retrieval_required=False)
    assert is_retrieval_adequate(set(), g)


def test_degenerate_empty_retrieved_set() -> None:
    g = _golden(required_doc_ids=["a"])
    assert not is_retrieval_adequate(set(), g)
