"""Tests for trajeval.adapters."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trajeval.adapters import (
    AgentAdapter,
    CallableAdapter,
    OpenAIToolCallAdapter,
    TrajectoryRecorder,
    parse_openai_messages,
)
from trajeval.types import AnswerStep, RetrievalStep, ThoughtStep, ToolStep, Trajectory

# ---------------------------------------------------------------------------
# CallableAdapter
# ---------------------------------------------------------------------------


def test_callable_adapter_wraps_a_plain_function() -> None:
    def agent(question: str) -> dict:
        return {
            "final_answer": "Paris",
            "steps": [{"step_type": "answer", "text": "Paris"}],
        }

    adapter = CallableAdapter(agent)
    assert isinstance(adapter, AgentAdapter)
    trajectory = adapter.run("What is the capital of France?")
    assert trajectory.question == "What is the capital of France?"
    assert trajectory.final_answer == "Paris"
    assert len(trajectory.steps) == 1


def test_callable_adapter_degenerate_no_steps() -> None:
    adapter = CallableAdapter(lambda q: {"final_answer": "4"})
    trajectory = adapter.run("What is 2+2?")
    assert trajectory.steps == []


def test_callable_adapter_invalid_shape_raises() -> None:
    adapter = CallableAdapter(
        lambda q: {"final_answer": "ok", "steps": [{"step_type": "not_real"}]}
    )
    with pytest.raises(ValidationError):
        adapter.run("q")


# ---------------------------------------------------------------------------
# parse_openai_messages / OpenAIToolCallAdapter
# ---------------------------------------------------------------------------


def _full_message_list() -> list[dict]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What year was the Eiffel Tower completed?"},
        {
            "role": "assistant",
            "content": "I should search for this.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search_docs",
                        "arguments": '{"query": "Eiffel Tower completion year"}',
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expr": "1+1"}'},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '[{"doc_id": "doc-42", "text": "Completed in 1889.", "score": 0.9}]',
        },
        {"role": "tool", "tool_call_id": "call_2", "content": "2"},
        {"role": "assistant", "content": "1889."},
    ]


def test_parse_openai_messages_full_trajectory() -> None:
    trajectory = parse_openai_messages(
        _full_message_list(),
        question="What year was the Eiffel Tower completed?",
        retrieval_tool_names={"search_docs"},
    )
    assert trajectory.final_answer == "1889."
    types = [type(s) for s in trajectory.steps]
    assert types == [ThoughtStep, RetrievalStep, ToolStep, AnswerStep]

    retrieval = trajectory.steps[1]
    assert isinstance(retrieval, RetrievalStep)
    assert retrieval.query == "Eiffel Tower completion year"
    assert retrieval.chunks[0].doc_id == "doc-42"
    assert retrieval.chunks[0].score == 0.9

    tool = trajectory.steps[2]
    assert isinstance(tool, ToolStep)
    assert tool.tool_name == "calculator"
    assert tool.result == 2  # JSON-decoded, not left as the string "2"


def test_parse_openai_messages_degenerate_direct_answer_no_tool_calls() -> None:
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    trajectory = parse_openai_messages(
        messages, question="What is 2+2?", retrieval_tool_names=set()
    )
    assert trajectory.final_answer == "4"
    assert len(trajectory.steps) == 1
    assert isinstance(trajectory.steps[0], AnswerStep)


def test_parse_openai_messages_missing_final_answer_raises() -> None:
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search_docs", "arguments": '{"query": "x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "[]"},
    ]
    with pytest.raises(ValueError, match="No final assistant message"):
        parse_openai_messages(messages, question="q", retrieval_tool_names={"search_docs"})


def test_parse_openai_messages_retrieval_call_missing_query_arg_raises() -> None:
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search_docs", "arguments": '{"not_query": "x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "[]"},
        {"role": "assistant", "content": "done"},
    ]
    with pytest.raises(ValueError, match="query"):
        parse_openai_messages(messages, question="q", retrieval_tool_names={"search_docs"})


def test_default_chunk_parser_handles_wrapped_and_scalar_shapes() -> None:
    # dict wrapping a list under "results"
    messages_wrapped = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search_docs", "arguments": '{"query": "x"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": '{"results": [{"id": "d1", "content": "hello"}]}',
        },
        {"role": "assistant", "content": "done"},
    ]
    trajectory = parse_openai_messages(
        messages_wrapped, question="q", retrieval_tool_names={"search_docs"}
    )
    retrieval = trajectory.steps[0]
    assert isinstance(retrieval, RetrievalStep)
    assert retrieval.chunks[0].doc_id == "d1"
    assert retrieval.chunks[0].text == "hello"

    # a scalar / unparseable result degrades to one chunk rather than raising
    messages_scalar = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search_docs", "arguments": '{"query": "x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "no results found"},
        {"role": "assistant", "content": "done"},
    ]
    trajectory2 = parse_openai_messages(
        messages_scalar, question="q", retrieval_tool_names={"search_docs"}
    )
    retrieval2 = trajectory2.steps[0]
    assert isinstance(retrieval2, RetrievalStep)
    assert len(retrieval2.chunks) == 1
    assert retrieval2.chunks[0].text == "no results found"


def test_openai_tool_call_adapter_wraps_a_message_producing_function() -> None:
    adapter = OpenAIToolCallAdapter(
        fn=lambda q: _full_message_list(),
        retrieval_tool_names={"search_docs"},
    )
    assert isinstance(adapter, AgentAdapter)
    trajectory = adapter.run("What year was the Eiffel Tower completed?")
    assert trajectory.final_answer == "1889."


# ---------------------------------------------------------------------------
# TrajectoryRecorder
# ---------------------------------------------------------------------------


def test_trajectory_recorder_happy_path() -> None:
    with TrajectoryRecorder("What is 2+2?", model="gpt-4o") as rec:
        rec.thought("Easy, no need to search.")
        rec.tool("calculator", args={"expr": "2+2"}, result=4)
        rec.answer("4")

    assert rec.trajectory is not None
    assert isinstance(rec.trajectory, Trajectory)
    assert rec.trajectory.final_answer == "4"
    assert rec.trajectory.metadata.model == "gpt-4o"
    assert [type(s) for s in rec.trajectory.steps] == [ThoughtStep, ToolStep, AnswerStep]


def test_trajectory_recorder_retrieve_accepts_dict_chunks() -> None:
    with TrajectoryRecorder("q") as rec:
        rec.retrieve(query="x", chunks=[{"doc_id": "d1", "text": "hi"}])
        rec.answer("a")
    assert rec.trajectory is not None
    step = rec.trajectory.steps[0]
    assert isinstance(step, RetrievalStep)
    assert step.chunks[0].doc_id == "d1"


def test_trajectory_recorder_degenerate_answer_only() -> None:
    with TrajectoryRecorder("q") as rec:
        rec.answer("a")
    assert rec.trajectory is not None
    assert len(rec.trajectory.steps) == 1


def test_trajectory_recorder_missing_answer_raises_on_exit() -> None:
    with pytest.raises(ValueError, match="without an answer"):
        with TrajectoryRecorder("q") as rec:
            rec.thought("thinking...")
    assert rec.trajectory is None


def test_trajectory_recorder_propagates_exception_without_building_trajectory() -> None:
    with pytest.raises(RuntimeError):
        with TrajectoryRecorder("q") as rec:
            rec.thought("thinking...")
            raise RuntimeError("agent crashed")
    assert rec.trajectory is None
