"""Reduces a judged metric's per-trajectory `MetricResult.details` to a
single categorical verdict string, so a human labeler can be asked to
reproduce exactly one decision instead of a whole per-claim or per-query
breakdown.

This is a deliberate simplification, not a hidden precision: `faithfulness`
scores per-claim and `query_quality` scores per-query, but calibration here
measures agreement on one trajectory-level summary of each. That's a
reasonable proxy for "does the judge broadly agree with a human here", not a
substitute for inspecting individual claim/query verdicts by hand — the
report's trace view is where that inspection happens.

Returns None when the metric produced no judged decision to calibrate for
this trajectory (metric didn't apply, or — for `recovery` — the outcome was
the deterministic `reformulated` branch, which never called the judge).
"""

from __future__ import annotations

from typing import cast

from pydantic import JsonValue

from trajeval.metrics.base import MetricResult

_Record = dict[str, JsonValue]


def judge_verdict(metric_name: str, result: MetricResult) -> str | None:
    if result.value is None:
        return None

    if metric_name == "recovery":
        outcome = result.details.get("outcome")
        if outcome in (None, "reformulated"):
            return None
        return str(outcome)

    if metric_name == "query_quality":
        raw_queries = result.details.get("queries")
        if not isinstance(raw_queries, list) or not raw_queries:
            return None
        queries = [cast(_Record, q) for q in raw_queries]
        qualities = [float(cast(float, q["judged_quality"])) for q in queries]
        mean_quality = sum(qualities) / len(qualities)
        return str(round(mean_quality))

    if metric_name == "faithfulness":
        raw_claims = result.details.get("claims")
        if not isinstance(raw_claims, list) or not raw_claims:
            return None
        claims = [cast(_Record, c) for c in raw_claims]
        all_supported = all(c["supported"] for c in claims)
        return "fully_supported" if all_supported else "not_fully_supported"

    return None
