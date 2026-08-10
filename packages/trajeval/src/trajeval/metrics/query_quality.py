"""query_quality: how good were the search queries the agent wrote?

Two components, reported **separately** — collapsing them into one number
would bury a deterministic signal inside a judged one, which this project's
design rules explicitly forbid:

- ``hit`` (deterministic): did this query's returned chunks include any doc
  id this golden record considers relevant? Pure set membership, no model
  involved.
- ``judged_quality`` (LLM-judged, 1-5 normalized to 0-1): is the query
  well-formed, specific, and free of conversational noise agents often
  leak into search calls ("um, I should probably look up...")?

Applies only to trajectories with at least one `RetrievalStep`.
"""

from __future__ import annotations

from typing import cast

from pydantic import JsonValue

from trajeval.judge.client import JudgeClient, extract_json
from trajeval.metrics.base import MetricResult
from trajeval.metrics.context import relevant_doc_ids, step_chunk_ids
from trajeval.types import GoldenRecord, RetrievalStep, Trajectory

QueryRecord = dict[str, JsonValue]

_QUALITY_PROMPT = """You are evaluating a search query issued by an AI agent inside a \
retrieval-augmented question-answering system.

Query: "{query}"

Rate this query's quality as a search query on a 1-5 scale:
5 = specific, well-formed, uses the key terms a real search index would need
3 = usable but vague, overly broad, or slightly malformed
1 = not a real search query (conversational filler, a sentence addressed to \
a person, or empty)

Respond with ONLY a JSON object, no other text: \
{{"score": <integer 1-5>, "reason": "<one sentence>"}}"""


class QueryQualityMetric:
    name = "query_quality"

    def __init__(self, judge: JudgeClient) -> None:
        self._judge = judge

    def score(self, trajectory: Trajectory, golden: GoldenRecord) -> MetricResult:
        retrievals = [s for s in trajectory.steps if isinstance(s, RetrievalStep)]
        if not retrievals:
            return MetricResult(metric_name=self.name, value=None, details={"queries": []})

        relevant = relevant_doc_ids(golden)
        per_query: list[QueryRecord] = []
        qualities: list[int] = []
        for step in retrievals:
            hit = bool(relevant) and bool(step_chunk_ids(step) & relevant)
            quality_1_5 = self._judge_quality(step.query)
            qualities.append(quality_1_5)
            per_query.append({"query": step.query, "hit": hit, "judged_quality": quality_1_5})

        mean_quality = sum(qualities) / len(qualities)
        value = (mean_quality - 1) / 4  # normalize 1-5 -> 0-1
        details: dict[str, JsonValue] = {"queries": cast(list[JsonValue], per_query)}
        return MetricResult(metric_name=self.name, value=value, details=details)

    def _judge_quality(self, query: str) -> int:
        response = self._judge.judge(_QUALITY_PROMPT.format(query=query))
        parsed = extract_json(response)
        score = int(parsed["score"])
        return max(1, min(5, score))

    def aggregate(self, results: list[MetricResult]) -> dict[str, JsonValue]:
        applicable = [r for r in results if r.value is not None]
        if not applicable:
            return {
                "total": len(results),
                "applicable": 0,
                "mean_judged_quality": None,
                "hit_rate": None,
            }

        all_queries = [
            cast(QueryRecord, q)
            for r in applicable
            for q in cast(list[JsonValue], r.details["queries"])
        ]
        hits = [bool(q["hit"]) for q in all_queries]
        return {
            "total": len(results),
            "applicable": len(applicable),
            "mean_judged_quality": sum(r.value for r in applicable if r.value is not None)
            / len(applicable),
            "hit_rate": sum(1 for h in hits if h) / len(hits) if hits else None,
            "total_queries": len(all_queries),
        }
