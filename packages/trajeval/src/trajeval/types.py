"""Core domain schema for trajeval.

Everything an evaluation run operates on lives here: the steps an agent takes,
the trajectory those steps form, and the golden record a trajectory is scored
against. Nothing in this module talks to an LLM, a retriever, or an agent
framework — it is pure data.

All models round-trip losslessly through ``model_dump_json()`` /
``model_validate_json()``. Both `Trajectory` and `GoldenRecord` carry a
`schema_version` so that a `RunResult` produced by an older version of this
library can be detected and migrated (or rejected) rather than silently
misread.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

# Bump this whenever a field is added, removed, or reinterpreted in a way that
# would change how an existing JSON file on disk should be read.
SCHEMA_VERSION = 1

# Tool call args/results are arbitrary but must still serialise losslessly to
# and from JSON. `pydantic.JsonValue` is a recursive JSON-safe type; we reuse
# it rather than hand-rolling one, which hits a recursion issue under 3.11's
# implicit forward-ref resolution (see pydantic/pydantic#9704).


class TrajevalModel(BaseModel):
    """Base class for all trajeval schema models.

    ``extra="forbid"`` is deliberate: a step or record with a field we don't
    recognise is far more likely to be a typo or a version mismatch than
    intentional, and we want that to fail loudly at parse time rather than
    silently drop data.
    """

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


class StepType(StrEnum):
    THOUGHT = "thought"
    RETRIEVAL = "retrieval"
    TOOL = "tool"
    ANSWER = "answer"


class RetrievedChunk(TrajevalModel):
    """One chunk returned by a single retrieval call."""

    doc_id: str
    text: str
    # Retriever scores are not comparable across backends (cosine similarity,
    # BM25, a reranker's logit...), so this is deliberately an opaque,
    # unbounded float rather than an assumed-normalised [0, 1] value.
    score: float | None = None
    rank: int | None = None


class ThoughtStep(TrajevalModel):
    """Reasoning text the agent emitted before acting."""

    step_type: Literal[StepType.THOUGHT] = StepType.THOUGHT
    text: str
    timestamp: datetime | None = None


class RetrievalStep(TrajevalModel):
    """A single search: the query issued and what came back."""

    step_type: Literal[StepType.RETRIEVAL] = StepType.RETRIEVAL
    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    timestamp: datetime | None = None


class ToolStep(TrajevalModel):
    """A non-retrieval tool call: name, arguments, and result."""

    step_type: Literal[StepType.TOOL] = StepType.TOOL
    tool_name: str
    args: dict[str, JsonValue] = Field(default_factory=dict)
    result: JsonValue = None
    timestamp: datetime | None = None


class AnswerStep(TrajevalModel):
    """The agent producing its final response."""

    step_type: Literal[StepType.ANSWER] = StepType.ANSWER
    text: str
    timestamp: datetime | None = None


# Discriminated union so that `model_validate_json` on a `Trajectory` routes
# each step dict to the right concrete model based on `step_type`, and so
# that mypy narrows on it after an `isinstance`/match check.
Step = Annotated[
    ThoughtStep | RetrievalStep | ToolStep | AnswerStep,
    Field(discriminator="step_type"),
]


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


class TrajectoryMetadata(TrajevalModel):
    """Run-level metadata: model, timing, token counts.

    ``extra="allow"`` here (overriding the base class) because callers will
    reasonably want to stash adapter-specific extras (e.g. a provider request
    id) without us having anticipated every field up front. The named fields
    below are the ones metrics and reports are allowed to depend on.
    """

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    timestamp: datetime | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None


class Trajectory(TrajevalModel):
    """The full record of an agent answering one question."""

    schema_version: int = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: uuid4().hex)
    # Links this trajectory to the GoldenRecord it should be scored against.
    # None is allowed so a Trajectory can be constructed/recorded standalone
    # (e.g. by TrajectoryRecorder, before the runner pairs it with a dataset).
    golden_id: str | None = None
    question: str
    final_answer: str
    steps: list[Step] = Field(default_factory=list)
    metadata: TrajectoryMetadata = Field(default_factory=TrajectoryMetadata)


# ---------------------------------------------------------------------------
# Golden record
# ---------------------------------------------------------------------------


class GoldenRecord(TrajevalModel):
    """Ground truth for one question."""

    schema_version: int = SCHEMA_VERSION
    id: str
    question: str
    reference_answer: str

    # AND-semantics: every id in this list must be retrieved somewhere in the
    # trajectory for retrieval to count as adequate.
    required_doc_ids: list[str] = Field(default_factory=list)
    # OR-semantics: retrieving any single one of these ids is sufficient on
    # its own. Mutually exclusive with required_doc_ids — see validator below.
    sufficient_doc_ids: list[str] = Field(default_factory=list)

    # False = answerable from the model's parametric knowledge; no retrieval
    # should have been necessary. This is what catches reflexive searching.
    retrieval_required: bool
    min_steps: int = Field(ge=0)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_doc_id_semantics(self) -> GoldenRecord:
        if self.required_doc_ids and self.sufficient_doc_ids:
            raise ValueError(
                "required_doc_ids and sufficient_doc_ids are mutually exclusive: "
                "a golden record expresses either AND-semantics (all of "
                "required_doc_ids must be retrieved) or OR-semantics (any one "
                "of sufficient_doc_ids is enough), never both."
            )
        if not self.retrieval_required and (self.required_doc_ids or self.sufficient_doc_ids):
            raise ValueError(
                "retrieval_required=False but doc ids were given. A record "
                "answerable from parametric knowledge should specify no "
                "required_doc_ids or sufficient_doc_ids."
            )
        return self
