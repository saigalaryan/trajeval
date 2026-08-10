"""The plugin seam between trajeval and an arbitrary agent.

trajeval never imports an agent framework. `AgentAdapter` is the one contract
everything else is built on: anything that can answer a question and hand
back a `Trajectory` satisfies it, whether it's a LangGraph graph, a
hand-rolled while loop, or a raw OpenAI-style message list parsed after the
fact.

Three ways in are provided:

- `CallableAdapter` — wrap a function that already returns a trajectory-shaped
  dict (the agent assembles its own step list).
- `OpenAIToolCallAdapter` — wrap a function that returns a raw OpenAI
  chat-completions message list; this module converts it.
- `TrajectoryRecorder` — a context manager for instrumenting an existing
  agent loop in place, a few lines at a time, without restructuring it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import TracebackType
from typing import Any, Literal, Protocol, runtime_checkable

from trajeval.types import (
    AnswerStep,
    RetrievalStep,
    RetrievedChunk,
    Step,
    ThoughtStep,
    ToolStep,
    Trajectory,
    TrajectoryMetadata,
)


@runtime_checkable
class AgentAdapter(Protocol):
    """Anything that can answer a question and return a Trajectory."""

    def run(self, question: str) -> Trajectory: ...


# ---------------------------------------------------------------------------
# CallableAdapter
# ---------------------------------------------------------------------------


class CallableAdapter:
    """Wraps a function returning a trajectory-shaped dict.

    `fn` receives the question and returns a mapping matching `Trajectory`'s
    fields (`final_answer`, `steps`, ...). `question` is filled in from the
    call if the mapping omits it, so the function doesn't need to repeat it.
    """

    def __init__(self, fn: Callable[[str], Mapping[str, Any]]) -> None:
        self._fn = fn

    def run(self, question: str) -> Trajectory:
        raw = dict(self._fn(question))
        raw.setdefault("question", question)
        return Trajectory.model_validate(raw)


# ---------------------------------------------------------------------------
# OpenAIToolCallAdapter
# ---------------------------------------------------------------------------


def _try_json_parse(value: Any) -> Any:
    """Best-effort JSON decode; returns the original value if it isn't JSON."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _default_chunk_parser(raw_result: Any) -> list[RetrievedChunk]:
    """Turn a retrieval tool's raw result into RetrievedChunks.

    Handles the shapes a retrieval tool commonly returns: a JSON-encoded (or
    already-parsed) list of chunk dicts, a dict wrapping that list under a
    `chunks`/`results` key, or a bare list of strings. Anything else becomes
    a single chunk holding the stringified result, so parsing never silently
    drops a result — it degrades to one low-fidelity chunk instead.

    Callers whose tool has a different result shape should pass their own
    `chunk_parser` to `OpenAIToolCallAdapter` rather than relying on this.
    """
    parsed = _try_json_parse(raw_result)
    if isinstance(parsed, dict):
        for key in ("chunks", "results", "documents"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break

    if not isinstance(parsed, list):
        return [RetrievedChunk(doc_id="chunk-0", text=str(parsed))]

    chunks: list[RetrievedChunk] = []
    for i, item in enumerate(parsed):
        if isinstance(item, dict):
            doc_id = str(item.get("doc_id", item.get("id", f"chunk-{i}")))
            text = str(item.get("text", item.get("content", item)))
            score = item.get("score")
            rank = item.get("rank", i + 1)
            chunks.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    text=text,
                    score=float(score) if score is not None else None,
                    rank=int(rank) if rank is not None else None,
                )
            )
        else:
            chunks.append(RetrievedChunk(doc_id=f"chunk-{i}", text=str(item), rank=i + 1))
    return chunks


def parse_openai_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    question: str,
    retrieval_tool_names: Iterable[str],
    query_arg_key: str = "query",
    chunk_parser: Callable[[Any], list[RetrievedChunk]] = _default_chunk_parser,
    metadata: Mapping[str, Any] | None = None,
) -> Trajectory:
    """Convert a raw OpenAI chat-completions message list into a Trajectory.

    Scope: single-question tool-calling loops — one `user` message, any
    number of `assistant`/`tool` messages, ending in an `assistant` message
    with no tool calls. `system` messages are ignored. Multi-turn
    conversations (several `user` messages) are out of scope; use
    `CallableAdapter` or `TrajectoryRecorder` for those.

    Retrieval tool calls (name in `retrieval_tool_names`) become
    `RetrievalStep`s: the query is read from argument `query_arg_key`, and
    the result is parsed into chunks by `chunk_parser`. Every other tool call
    becomes a plain `ToolStep`. Assistant text preceding a tool call becomes
    a `ThoughtStep`; assistant text with no tool call is the final answer.
    """
    retrieval_names = set(retrieval_tool_names)
    tool_results: dict[str, Any] = {
        m["tool_call_id"]: m.get("content") for m in messages if m.get("role") == "tool"
    }

    steps: list[Step] = []
    final_answer: str | None = None

    for message in messages:
        role = message.get("role")
        if role in ("system", "user"):
            continue
        if role != "assistant":
            continue  # "tool" messages are consumed via tool_results above

        content = message.get("content")
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            if content:
                final_answer = str(content)
                steps.append(AnswerStep(text=final_answer))
            continue

        if content:
            steps.append(ThoughtStep(text=str(content)))

        for call in tool_calls:
            fn = call["function"]
            name = fn["name"]
            args = _try_json_parse(fn.get("arguments", "{}"))
            if not isinstance(args, dict):
                args = {"raw": args}
            result = _try_json_parse(tool_results.get(call["id"]))

            if name in retrieval_names:
                query = args.get(query_arg_key)
                if query is None:
                    raise ValueError(
                        f"Retrieval tool call {name!r} has no {query_arg_key!r} "
                        f"argument (args={args!r}). Pass a different "
                        "query_arg_key if this tool names it something else."
                    )
                steps.append(RetrievalStep(query=str(query), chunks=chunk_parser(result)))
            else:
                steps.append(ToolStep(tool_name=name, args=args, result=result))

    if final_answer is None:
        raise ValueError(
            "No final assistant message without tool calls was found — the "
            "message list doesn't end in an answer, so no Trajectory.final_answer "
            "can be determined."
        )

    return Trajectory(
        question=question,
        final_answer=final_answer,
        steps=steps,
        metadata=TrajectoryMetadata.model_validate(metadata) if metadata else TrajectoryMetadata(),
    )


class OpenAIToolCallAdapter:
    """Wraps a function that runs an agent and returns raw OpenAI messages.

    `fn` receives the question and returns the message list produced by the
    chat-completions loop (system/user/assistant/tool messages, with
    `tool_calls` on assistant messages). This adapter converts that into a
    `Trajectory` via `parse_openai_messages`, so an agent that already speaks
    the standard tool-calling format needs no instrumentation at all.
    """

    def __init__(
        self,
        fn: Callable[[str], Sequence[Mapping[str, Any]]],
        retrieval_tool_names: Iterable[str],
        query_arg_key: str = "query",
        chunk_parser: Callable[[Any], list[RetrievedChunk]] = _default_chunk_parser,
    ) -> None:
        self._fn = fn
        self._retrieval_tool_names = set(retrieval_tool_names)
        self._query_arg_key = query_arg_key
        self._chunk_parser = chunk_parser

    def run(self, question: str) -> Trajectory:
        messages = self._fn(question)
        return parse_openai_messages(
            messages,
            question=question,
            retrieval_tool_names=self._retrieval_tool_names,
            query_arg_key=self._query_arg_key,
            chunk_parser=self._chunk_parser,
        )


# ---------------------------------------------------------------------------
# TrajectoryRecorder
# ---------------------------------------------------------------------------


class TrajectoryRecorder:
    """Context manager for instrumenting an existing agent loop in place.

    ::

        with TrajectoryRecorder(question) as rec:
            rec.thought("I should look this up.")
            rec.retrieve(query="...", chunks=[{"doc_id": "d1", "text": "..."}])
            rec.tool("calculator", args={"expr": "1+1"}, result=2)
            rec.answer("42")
        trajectory = rec.trajectory

    `rec.trajectory` is only populated once the `with` block exits normally
    and `answer()` was called; an exception propagating out of the block
    leaves it `None` rather than building a trajectory from a run that never
    finished.
    """

    def __init__(self, question: str, **metadata: Any) -> None:
        self._question = question
        self._metadata_kwargs = metadata
        self._steps: list[Step] = []
        self._final_answer: str | None = None
        self.trajectory: Trajectory | None = None

    def thought(self, text: str) -> None:
        self._steps.append(ThoughtStep(text=text))

    def retrieve(
        self, query: str, chunks: Sequence[RetrievedChunk | Mapping[str, Any]] = ()
    ) -> None:
        parsed = [
            c if isinstance(c, RetrievedChunk) else RetrievedChunk.model_validate(c) for c in chunks
        ]
        self._steps.append(RetrievalStep(query=query, chunks=parsed))

    def tool(
        self, tool_name: str, args: Mapping[str, Any] | None = None, result: Any = None
    ) -> None:
        self._steps.append(ToolStep(tool_name=tool_name, args=dict(args or {}), result=result))

    def answer(self, text: str) -> None:
        self._steps.append(AnswerStep(text=text))
        self._final_answer = text

    def __enter__(self) -> TrajectoryRecorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        if exc_type is not None:
            return False  # propagate; don't build a trajectory from a failed run
        if self._final_answer is None:
            raise ValueError(
                "TrajectoryRecorder exited without an answer — call "
                "rec.answer(...) before the `with` block ends."
            )
        self.trajectory = Trajectory(
            question=self._question,
            final_answer=self._final_answer,
            steps=self._steps,
            metadata=TrajectoryMetadata.model_validate(self._metadata_kwargs),
        )
        return False
