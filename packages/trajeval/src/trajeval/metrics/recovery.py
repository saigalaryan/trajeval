"""recovery: given a retrieval that came back with nothing relevant, what did
the agent do next?

Only applies to trajectories with at least one "bad" retrieval — a
`RetrievalStep` whose returned chunks include none of the golden record's
relevant doc ids. Trajectories with no bad retrieval get
`MetricResult(value=None)`: not applicable, distinct from scoring 0.

Given a bad retrieval happened, the agent did one of three things:

- **reformulated** — issued another, meaningfully different query
  afterward. Fully deterministic: another `RetrievalStep` whose query has a
  normalized edit distance above the loop threshold from the bad one.
- **answered_from_bad_context** — answered anyway, treating the irrelevant
  chunks as if they supported the answer. The most dangerous production
  failure mode this project's brief calls out, and almost nothing else
  measures it.
- **correctly_abstained** — declined to answer, or answered without relying
  on the bad context (e.g. from parametric knowledge, with the limitation
  stated).

Distinguishing "answered from bad context" from "correctly abstained" is a
judgment call about what the final answer actually asserts and how it
relates to the bad chunks — that part is judged. Whether the agent
reformulated is not — it's a fact about the step sequence.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import JsonValue

from trajeval.judge.client import JudgeClient, extract_json
from trajeval.metrics._text import normalized_edit_distance
from trajeval.metrics.base import MetricResult
from trajeval.metrics.context import relevant_doc_ids, step_chunk_ids
from trajeval.metrics.trajectory_efficiency import LOOP_DISTANCE_THRESHOLD
from trajeval.types import GoldenRecord, RetrievalStep, Trajectory

_ABSTENTION_PROMPT = """You are reviewing an AI agent's answer after a search that returned \
irrelevant results.

Question: {question}

The search returned these chunks, none of which are actually relevant to the question:
---
{bad_context}
---

The agent's final answer was:
---
{final_answer}
---

Does the final answer rely on or assert claims from the irrelevant search \
results above (or otherwise state something as fact that the search did \
not actually support), or does it correctly avoid doing so — for example \
by answering from general knowledge with the limitation acknowledged, or \
by declining to answer?

Respond with ONLY a JSON object, no other text: \
{{"outcome": "answered_from_bad_context" | "correctly_abstained", "reason": "<one sentence>"}}"""


class RecoveryOutcome(StrEnum):
    REFORMULATED = "reformulated"
    ANSWERED_FROM_BAD_CONTEXT = "answered_from_bad_context"
    CORRECTLY_ABSTAINED = "correctly_abstained"


def _is_bad_retrieval(step: RetrievalStep, relevant: set[str]) -> bool:
    """A retrieval is "bad" if it returned nothing relevant. If the golden
    record has no relevant doc ids at all (retrieval_required=False, so
    nothing was actually needed), no retrieval can be judged "bad" by this
    definition — that failure mode is over-retrieval, not recovery."""
    if not relevant:
        return False
    return not (step_chunk_ids(step) & relevant)


class RecoveryMetric:
    name = "recovery"

    def __init__(self, judge: JudgeClient) -> None:
        self._judge = judge

    def score(self, trajectory: Trajectory, golden: GoldenRecord) -> MetricResult:
        relevant = relevant_doc_ids(golden)
        retrievals = [
            (i, s) for i, s in enumerate(trajectory.steps) if isinstance(s, RetrievalStep)
        ]
        bad = next(((i, s) for i, s in retrievals if _is_bad_retrieval(s, relevant)), None)

        if bad is None:
            return MetricResult(metric_name=self.name, value=None, details={})

        bad_index, bad_step = bad
        later_queries = [s.query for i, s in retrievals if i > bad_index]
        reformulated = any(
            normalized_edit_distance(q, bad_step.query) >= LOOP_DISTANCE_THRESHOLD
            for q in later_queries
        )

        if reformulated:
            outcome = RecoveryOutcome.REFORMULATED
        else:
            outcome = self._judge_abstention(trajectory, golden, bad_step)

        value = 0.0 if outcome == RecoveryOutcome.ANSWERED_FROM_BAD_CONTEXT else 1.0
        return MetricResult(
            metric_name=self.name,
            value=value,
            details={"outcome": outcome.value, "bad_retrieval_step": bad_index},
        )

    def _judge_abstention(
        self, trajectory: Trajectory, golden: GoldenRecord, bad_step: RetrievalStep
    ) -> RecoveryOutcome:
        bad_context = (
            "\n".join(f"[{c.doc_id}] {c.text}" for c in bad_step.chunks) or "(no chunks returned)"
        )
        prompt = _ABSTENTION_PROMPT.format(
            question=golden.question, bad_context=bad_context, final_answer=trajectory.final_answer
        )
        response = self._judge.judge(prompt)
        parsed = extract_json(response)
        outcome_str = parsed["outcome"]
        if outcome_str == "answered_from_bad_context":
            return RecoveryOutcome.ANSWERED_FROM_BAD_CONTEXT
        return RecoveryOutcome.CORRECTLY_ABSTAINED

    def aggregate(self, results: list[MetricResult]) -> dict[str, JsonValue]:
        applicable = [r for r in results if r.value is not None]
        if not applicable:
            return {
                "total": len(results),
                "applicable": 0,
                "reformulated": 0,
                "answered_from_bad_context": 0,
                "correctly_abstained": 0,
                "answered_from_bad_context_rate": None,
            }

        counts = {outcome.value: 0 for outcome in RecoveryOutcome}
        for r in applicable:
            outcome = r.details["outcome"]
            assert isinstance(outcome, str)
            counts[outcome] += 1

        return {
            "total": len(results),
            "applicable": len(applicable),
            "reformulated": counts[RecoveryOutcome.REFORMULATED.value],
            "answered_from_bad_context": counts[RecoveryOutcome.ANSWERED_FROM_BAD_CONTEXT.value],
            "correctly_abstained": counts[RecoveryOutcome.CORRECTLY_ABSTAINED.value],
            "answered_from_bad_context_rate": counts[
                RecoveryOutcome.ANSWERED_FROM_BAD_CONTEXT.value
            ]
            / len(applicable),
        }
