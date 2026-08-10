"""trajectory_efficiency: did the agent take the optimal number of steps?

`value = min_steps / actual_steps`, clipped to 1.0 — a trajectory that took
exactly `min_steps` scores 1.0; one that took more scores lower; one that
(implausibly) took fewer than `min_steps` also clips to 1.0 rather than
scoring "better than perfect" — `min_steps` is a floor on what's needed to
answer correctly, not a target to beat.

Loops are flagged separately from the efficiency score: two `RetrievalStep`s
with near-identical queries (normalized edit distance below
`LOOP_DISTANCE_THRESHOLD`) and no `ToolStep` between them signal the agent
repeating itself rather than making progress. `ThoughtStep`s between two
near-identical retrievals don't break the loop — an agent "thinking out
loud" between two identical searches is still looping.

Fully deterministic — no LLM involved.
"""

from __future__ import annotations

from pydantic import JsonValue

from trajeval.metrics._text import normalized_edit_distance
from trajeval.metrics.base import MetricResult
from trajeval.types import GoldenRecord, RetrievalStep, ToolStep, Trajectory

LOOP_DISTANCE_THRESHOLD = 0.15


def detect_loops(trajectory: Trajectory) -> list[tuple[int, int]]:
    """Return (earlier_index, later_index) pairs of retrieval steps whose
    queries are near-identical with no intervening tool call."""
    pairs: list[tuple[int, int]] = []
    prev_index: int | None = None
    prev_query: str | None = None

    for i, step in enumerate(trajectory.steps):
        if isinstance(step, RetrievalStep):
            if prev_index is not None:
                between = trajectory.steps[prev_index + 1 : i]
                no_tool_between = not any(isinstance(s, ToolStep) for s in between)
                if no_tool_between and prev_query is not None:
                    if normalized_edit_distance(step.query, prev_query) < LOOP_DISTANCE_THRESHOLD:
                        pairs.append((prev_index, i))
            prev_index = i
            prev_query = step.query
        elif isinstance(step, ToolStep):
            # a tool call resets the "no intervening tool call" window
            prev_index = None
            prev_query = None

    return pairs


class TrajectoryEfficiencyMetric:
    name = "trajectory_efficiency"

    def score(self, trajectory: Trajectory, golden: GoldenRecord) -> MetricResult:
        actual_steps = len(trajectory.steps)
        loops = detect_loops(trajectory)

        if actual_steps == 0:
            # Nothing to measure efficiency of.
            return MetricResult(
                metric_name=self.name,
                value=None,
                details={"loop_count": 0, "has_loop": False, "actual_steps": 0},
            )

        value = min(golden.min_steps / actual_steps, 1.0)
        return MetricResult(
            metric_name=self.name,
            value=value,
            details={
                "loop_count": len(loops),
                "has_loop": len(loops) > 0,
                "actual_steps": actual_steps,
                "min_steps": golden.min_steps,
            },
        )

    def aggregate(self, results: list[MetricResult]) -> dict[str, JsonValue]:
        applicable = [r for r in results if r.value is not None]
        if not applicable:
            return {
                "total": len(results),
                "applicable": 0,
                "mean_efficiency": None,
                "loop_rate": None,
            }

        loop_count = sum(1 for r in applicable if r.details.get("has_loop"))
        return {
            "total": len(results),
            "applicable": len(applicable),
            "mean_efficiency": sum(r.value for r in applicable if r.value is not None)
            / len(applicable),
            "loop_rate": loop_count / len(applicable),
        }
