"""retrieval_necessity: did the agent search exactly when it should have?

Fully deterministic — no LLM involved, on purpose: this is the metric that
proves the harness can classify agent behaviour correctly before any judged
metric is trusted to.

Every trajectory is classified into one of four outcomes by crossing "did a
retrieval happen" against the golden record's `retrieval_required` flag:

| retrieved | retrieval_required | outcome          |
|-----------|---------------------|-------------------|
| yes       | yes                 | correct_search    |
| no        | no                  | correct_skip      |
| yes       | no                  | over_retrieval    |
| no        | yes                 | under_retrieval   |

The aggregate reports the four counts, a `necessity_score` (fraction
correct), and the over-/under-retrieval rates *separately* — averaging them
into one number hides that wasting tokens on unneeded search and
hallucinating from missing context are very different failure costs.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import JsonValue

from trajeval.metrics.base import MetricResult
from trajeval.types import GoldenRecord, RetrievalStep, Trajectory


class RetrievalOutcome(StrEnum):
    CORRECT_SEARCH = "correct_search"
    CORRECT_SKIP = "correct_skip"
    OVER_RETRIEVAL = "over_retrieval"
    UNDER_RETRIEVAL = "under_retrieval"


_CORRECT_OUTCOMES = frozenset({RetrievalOutcome.CORRECT_SEARCH, RetrievalOutcome.CORRECT_SKIP})


def _did_retrieve(trajectory: Trajectory) -> bool:
    """True if any step in the trajectory is a retrieval — including a
    RetrievalStep that came back with zero chunks. The question this metric
    answers is whether the agent *decided to search*, not whether the search
    succeeded; a search that returned nothing is still a search.
    """
    return any(isinstance(step, RetrievalStep) for step in trajectory.steps)


def classify(trajectory: Trajectory, golden: GoldenRecord) -> RetrievalOutcome:
    """Classify a single trajectory against its golden record."""
    retrieved = _did_retrieve(trajectory)
    required = golden.retrieval_required
    if retrieved and required:
        return RetrievalOutcome.CORRECT_SEARCH
    if not retrieved and not required:
        return RetrievalOutcome.CORRECT_SKIP
    if retrieved and not required:
        return RetrievalOutcome.OVER_RETRIEVAL
    return RetrievalOutcome.UNDER_RETRIEVAL


class RetrievalNecessityMetric:
    """See module docstring. Deterministic — no judge, no calibration needed."""

    name = "retrieval_necessity"

    def score(self, trajectory: Trajectory, golden: GoldenRecord) -> MetricResult:
        outcome = classify(trajectory, golden)
        value = 1.0 if outcome in _CORRECT_OUTCOMES else 0.0
        return MetricResult(metric_name=self.name, value=value, details={"outcome": outcome.value})

    def aggregate(self, results: list[MetricResult]) -> dict[str, JsonValue]:
        total = len(results)
        if total == 0:
            return {
                "total": 0,
                "correct_search": 0,
                "correct_skip": 0,
                "over_retrieval": 0,
                "under_retrieval": 0,
                "necessity_score": None,
                "over_retrieval_rate": None,
                "under_retrieval_rate": None,
            }

        counts = {outcome.value: 0 for outcome in RetrievalOutcome}
        for result in results:
            outcome = result.details["outcome"]
            if not isinstance(outcome, str) or outcome not in counts:
                raise ValueError(
                    f"RetrievalNecessityMetric.aggregate received a result whose "
                    f"details['outcome'] isn't one it produced: {outcome!r}"
                )
            counts[outcome] += 1

        correct = sum(counts[o.value] for o in _CORRECT_OUTCOMES)
        return {
            "total": total,
            "correct_search": counts[RetrievalOutcome.CORRECT_SEARCH.value],
            "correct_skip": counts[RetrievalOutcome.CORRECT_SKIP.value],
            "over_retrieval": counts[RetrievalOutcome.OVER_RETRIEVAL.value],
            "under_retrieval": counts[RetrievalOutcome.UNDER_RETRIEVAL.value],
            "necessity_score": correct / total,
            "over_retrieval_rate": counts[RetrievalOutcome.OVER_RETRIEVAL.value] / total,
            "under_retrieval_rate": counts[RetrievalOutcome.UNDER_RETRIEVAL.value] / total,
        }
