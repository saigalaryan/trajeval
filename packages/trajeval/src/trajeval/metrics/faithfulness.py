"""faithfulness: is every claim in the final answer supported by retrieved context?

Two judge calls, never one holistic score — a single "is this faithful?"
rating correlates poorly with human judgement (per this project's brief):

1. **Decompose** the final answer into atomic, independently-checkable claims.
2. **Verify** each claim against the retrieved context, in one batched call
   (cheaper than one call per claim, and the verdicts are still reported
   per-claim, not merged).

`value` is the fraction of claims judged supported. `details.claims` carries
every claim with its individual verdict — that per-claim breakdown, not the
fraction, is the point: it's what tells someone debugging their agent
exactly which sentence went unsupported.

Only applies when the trajectory retrieved at least one chunk (with no
context at all, "faithfulness to context" isn't a meaningful question —
that's what `retrieval_necessity` and `recovery` are for) and the answer
decomposes into at least one claim.
"""

from __future__ import annotations

from typing import cast

from pydantic import JsonValue

from trajeval.judge.client import JudgeClient, extract_json
from trajeval.metrics.base import MetricResult
from trajeval.types import GoldenRecord, RetrievalStep, Trajectory

ClaimVerdict = dict[str, JsonValue]

_DECOMPOSE_PROMPT = """Break the following answer into a list of atomic, independently \
checkable factual claims. Skip hedges, pleasantries, and meta-commentary \
("I found that...") — extract only the substantive claims.

Answer:
---
{answer}
---

Respond with ONLY a JSON array of strings, no other text. If the answer \
contains no checkable factual claims (e.g. it's a pure abstention like \
"I don't know"), respond with an empty array: []"""

_VERIFY_PROMPT = """You are checking whether each claim below is supported by the \
provided context. A claim is "supported" only if the context directly \
states it or clearly implies it — not if it merely doesn't contradict it.

Context:
---
{context}
---

Claims (respond to every one, in the same order):
{claims_list}

Respond with ONLY a JSON array, no other text, one object per claim in \
order: [{{"claim": "<the claim text>", "supported": true|false, "reason": \
"<one sentence>"}}, ...]"""


class FaithfulnessMetric:
    name = "faithfulness"

    def __init__(self, judge: JudgeClient) -> None:
        self._judge = judge

    def score(self, trajectory: Trajectory, golden: GoldenRecord) -> MetricResult:
        chunks = [c for s in trajectory.steps if isinstance(s, RetrievalStep) for c in s.chunks]
        if not chunks:
            return MetricResult(metric_name=self.name, value=None, details={})

        claims = self._decompose(trajectory.final_answer)
        if not claims:
            return MetricResult(metric_name=self.name, value=None, details={"claims": []})

        context_text = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in chunks)
        verdicts = self._verify(claims, context_text)

        supported = sum(1 for v in verdicts if v["supported"])
        value = supported / len(verdicts)
        details: dict[str, JsonValue] = {"claims": cast(list[JsonValue], verdicts)}
        return MetricResult(metric_name=self.name, value=value, details=details)

    def _decompose(self, answer: str) -> list[str]:
        response = self._judge.judge(_DECOMPOSE_PROMPT.format(answer=answer))
        parsed = extract_json(response)
        if not isinstance(parsed, list):
            raise ValueError(f"expected a JSON array of claims, got: {parsed!r}")
        return [str(c) for c in parsed]

    def _verify(self, claims: list[str], context_text: str) -> list[dict[str, JsonValue]]:
        claims_list = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
        response = self._judge.judge(
            _VERIFY_PROMPT.format(context=context_text, claims_list=claims_list)
        )
        parsed = extract_json(response)
        if not isinstance(parsed, list):
            raise ValueError(f"expected a JSON array of verdicts, got: {parsed!r}")
        return [dict(v) for v in parsed]

    def aggregate(self, results: list[MetricResult]) -> dict[str, JsonValue]:
        applicable = [r for r in results if r.value is not None]
        if not applicable:
            return {
                "total": len(results),
                "applicable": 0,
                "mean_faithfulness": None,
                "total_claims": 0,
                "unsupported_claims": 0,
            }

        all_claims = [
            cast(ClaimVerdict, c)
            for r in applicable
            for c in cast(list[JsonValue], r.details.get("claims", []))
        ]
        unsupported = sum(1 for c in all_claims if not c["supported"])
        return {
            "total": len(results),
            "applicable": len(applicable),
            "mean_faithfulness": sum(r.value for r in applicable if r.value is not None)
            / len(applicable),
            "total_claims": len(all_claims),
            "unsupported_claims": unsupported,
        }
