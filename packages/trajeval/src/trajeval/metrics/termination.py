"""termination: did the agent stop once the retrieved context was sufficient?

Compares the step index at which accumulated retrieved context first became
"adequate" (see `trajeval.metrics.context.is_retrieval_adequate` — handles
both the AND-semantics `required_doc_ids` and OR-semantics
`sufficient_doc_ids` golden-record modes) against the step index of the
`AnswerStep`. `excess_steps` is the gap: steps spent after the agent already
had what it needed.

Note the sign convention here is the opposite of every other metric in this
package: for `termination`, **lower `value` is better** (fewer excess
steps), whereas `retrieval_necessity`/`trajectory_efficiency`/etc. all use
"higher is better, 1.0 is best". This is deliberate — "mean excess steps" is
what the brief asks the aggregate to report, and forcing it onto a 0-1
"higher is better" scale would make the number harder to interpret, not
easier. Read `value` as "excess steps for this trajectory," not as a score.

Only applies when `retrieval_required` is true, at least one retrieval step
occurred, context became adequate at some point, and the agent answered —
absent any of those, there's no "became sufficient" moment to measure from.

Fully deterministic — no LLM involved.
"""

from __future__ import annotations

from pydantic import JsonValue

from trajeval.metrics.base import MetricResult
from trajeval.metrics.context import is_retrieval_adequate, step_chunk_ids
from trajeval.types import AnswerStep, GoldenRecord, RetrievalStep, Trajectory


class TerminationMetric:
    name = "termination"

    def score(self, trajectory: Trajectory, golden: GoldenRecord) -> MetricResult:
        if not golden.retrieval_required:
            return MetricResult(metric_name=self.name, value=None, details={})

        retrieved_so_far: set[str] = set()
        sufficient_at: int | None = None
        answer_at: int | None = None

        for i, step in enumerate(trajectory.steps):
            if isinstance(step, RetrievalStep):
                retrieved_so_far |= step_chunk_ids(step)
                if sufficient_at is None and is_retrieval_adequate(retrieved_so_far, golden):
                    sufficient_at = i
            elif isinstance(step, AnswerStep) and answer_at is None:
                answer_at = i

        if sufficient_at is None or answer_at is None:
            return MetricResult(metric_name=self.name, value=None, details={})

        # Steps strictly between "became sufficient" and "answered" — an
        # agent that answers on the very next step after sufficiency has
        # zero excess, not one.
        excess = max(0, answer_at - sufficient_at - 1)
        return MetricResult(
            metric_name=self.name,
            value=float(excess),
            details={"sufficient_at_step": sufficient_at, "answered_at_step": answer_at},
        )

    def aggregate(self, results: list[MetricResult]) -> dict[str, JsonValue]:
        applicable = [r for r in results if r.value is not None]
        if not applicable:
            return {"total": len(results), "applicable": 0, "mean_excess_steps": None}

        return {
            "total": len(results),
            "applicable": len(applicable),
            "mean_excess_steps": sum(r.value for r in applicable if r.value is not None)
            / len(applicable),
        }
