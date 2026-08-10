"""Shared interface every trajeval metric implements.

A metric produces two things: a per-trajectory result (what happened on this
one trajectory) and an aggregate over a whole run (what that means across the
dataset). The split matters — see `retrieval_necessity`'s over/under
distinction — because a single averaged number regularly hides which failure
mode is actually happening.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, JsonValue

from trajeval.types import GoldenRecord, Trajectory


class MetricResult(BaseModel):
    """One metric's verdict on one trajectory."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str
    # The metric's headline number for this trajectory, on a scale it
    # defines itself. None means the metric doesn't apply to this trajectory
    # at all (e.g. `recovery` when no bad retrieval occurred) — distinct from
    # 0.0, which means "applied, and scored the worst possible".
    value: float | None
    details: dict[str, JsonValue] = {}


@runtime_checkable
class Metric(Protocol):
    """A deterministic or judged scorer, pluggable into the runner."""

    name: str

    def score(self, trajectory: Trajectory, golden: GoldenRecord) -> MetricResult: ...

    def aggregate(self, results: list[MetricResult]) -> dict[str, JsonValue]: ...
