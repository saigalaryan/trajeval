"""Static, self-contained HTML report — inline CSS, no build step, no
external assets. What people paste into pull requests.

Shows: aggregate scores, per-tag breakdown, calibration state per metric,
cost summary, and the worst trajectories with full step-by-step traces. The
trace view matters more than the numbers — someone debugging their agent
needs to see exactly which query failed and what came back.
"""

from __future__ import annotations

import html as html_lib
from typing import Any

from trajeval.judge.client import FakeJudgeClient
from trajeval.metrics.base import Metric
from trajeval.metrics.faithfulness import FaithfulnessMetric
from trajeval.metrics.query_quality import QueryQualityMetric
from trajeval.metrics.recovery import RecoveryMetric
from trajeval.metrics.retrieval_necessity import RetrievalNecessityMetric
from trajeval.metrics.termination import TerminationMetric
from trajeval.metrics.trajectory_efficiency import TrajectoryEfficiencyMetric
from trajeval.results import RunResult, TrajectoryResult
from trajeval.types import AnswerStep, RetrievalStep, ThoughtStep, ToolStep

_METRIC_CLASSES: dict[str, type[Metric]] = {
    "retrieval_necessity": RetrievalNecessityMetric,
    "trajectory_efficiency": TrajectoryEfficiencyMetric,
    "termination": TerminationMetric,
    "query_quality": QueryQualityMetric,
    "recovery": RecoveryMetric,
    "faithfulness": FaithfulnessMetric,
}
_JUDGED_METRIC_NAMES = frozenset({"query_quality", "recovery", "faithfulness"})


def _metric_instance(name: str) -> Metric | None:
    """A throwaway metric instance, used only for its `.aggregate()` — a
    report re-slices existing `MetricResult`s by tag, it never re-judges
    anything, so a judged metric's judge is never actually called here.
    """
    cls = _METRIC_CLASSES.get(name)
    if cls is None:
        return None
    if name in _JUDGED_METRIC_NAMES:
        # type[Metric]'s synthesized signature (from the Protocol) takes no
        # constructor args; the concrete judged-metric classes take a judge.
        return cls(FakeJudgeClient({}))  # type: ignore[call-arg]
    return cls()


def _per_tag_breakdown(run_result: RunResult) -> dict[str, dict[str, dict[str, Any]]]:
    """tag -> metric_name -> aggregate() output, over trajectories carrying that tag."""
    tags = sorted({tag for tr in run_result.trajectory_results for tag in tr.tags})
    breakdown: dict[str, dict[str, dict[str, Any]]] = {}
    for tag in tags:
        subset = [tr for tr in run_result.trajectory_results if tag in tr.tags]
        breakdown[tag] = {}
        for metric_name in run_result.metadata.metric_names:
            instance = _metric_instance(metric_name)
            if instance is None:
                continue
            results = [
                tr.metric_results[metric_name] for tr in subset if metric_name in tr.metric_results
            ]
            breakdown[tag][metric_name] = instance.aggregate(results)
    return breakdown


def _worst_trajectories(
    run_result: RunResult, primary_metric: str | None = None, n: int = 10
) -> list[TrajectoryResult]:
    """The `n` trajectories with the lowest score on `primary_metric`.
    Trajectories where the metric didn't apply (value=None) sort last, not
    first — "not applicable" isn't "worst". Errored trajectories are
    excluded — there's no score to be worst at.
    """
    metric_name = primary_metric or next(iter(run_result.metadata.metric_names), None)
    if metric_name is None:
        return []

    scored = [tr for tr in run_result.trajectory_results if tr.error is None]

    def sort_key(tr: TrajectoryResult) -> tuple[int, float]:
        result = tr.metric_results.get(metric_name)
        if result is None or result.value is None:
            return (1, 0.0)
        return (0, result.value)

    scored.sort(key=sort_key)
    return scored[:n]


def _esc(value: Any) -> str:
    return html_lib.escape(str(value))


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return _esc(value)


def _render_aggregate_table(run_result: RunResult) -> str:
    rows = []
    for metric_name, agg in run_result.aggregate_scores.items():
        cal = run_result.calibration.get(metric_name)
        badge = ""
        if metric_name in _JUDGED_METRIC_NAMES:
            if cal is None or not cal.is_calibrated:
                badge = '<span class="badge badge-warn">uncalibrated</span>'
            else:
                badge = f'<span class="badge badge-ok">κ={cal.kappa:.2f} (n={cal.n_labels})</span>'
        headers = [k for k in agg if k != "total"]
        rows.append(
            f"<tr><th>{_esc(metric_name)} {badge}</th>"
            + "".join(f"<td>{_esc(h)}<br><strong>{_fmt(agg[h])}</strong></td>" for h in headers)
            + "</tr>"
        )
    return "<table class='agg-table'>" + "".join(rows) + "</table>"


def _render_tag_breakdown(run_result: RunResult) -> str:
    breakdown = _per_tag_breakdown(run_result)
    if not breakdown:
        return "<p><em>No tags in this dataset.</em></p>"
    sections = []
    for tag, metrics in breakdown.items():
        rows = "".join(
            f"<tr><td>{_esc(metric_name)}</td><td>{_esc(agg)}</td></tr>"
            for metric_name, agg in metrics.items()
        )
        sections.append(f"<h4>{_esc(tag)}</h4><table class='tag-table'>{rows}</table>")
    return "".join(sections)


def _render_cost_summary(cost_summary: dict[str, dict[str, Any]] | None) -> str:
    if not cost_summary:
        return "<p><em>No judge cost data recorded for this run.</em></p>"
    rows = "".join(
        f"<tr><td>{_esc(label)}</td><td>{_fmt(b.get('calls'))}</td>"
        f"<td>{_fmt(b.get('input_tokens'))}</td><td>{_fmt(b.get('output_tokens'))}</td>"
        f"<td>{'$' + _fmt(b['cost_usd']) if b.get('cost_usd') is not None else 'unknown'}</td></tr>"
        for label, b in cost_summary.items()
    )
    return (
        "<table class='cost-table'><tr><th>Label</th><th>Calls</th><th>Input tok</th>"
        f"<th>Output tok</th><th>Cost</th></tr>{rows}</table>"
    )


def _render_step(step: Any, index: int) -> str:
    idx = f"<span class='step-idx'>[{index}]</span>"
    if isinstance(step, ThoughtStep):
        return f"<div class='step step-thought'>{idx} thought: {_esc(step.text)}</div>"
    if isinstance(step, RetrievalStep):
        chunks = "".join(
            f"<li><code>{_esc(c.doc_id)}</code> (score={_fmt(c.score)}): {_esc(c.text[:300])}</li>"
            for c in step.chunks
        )
        chunks_html = chunks or "<li><em>no chunks</em></li>"
        return (
            f"<div class='step step-retrieval'>{idx} retrieval: "
            f"<code>{_esc(step.query)}</code><ul>{chunks_html}</ul></div>"
        )
    if isinstance(step, ToolStep):
        return (
            f"<div class='step step-tool'>{idx} tool: <code>{_esc(step.tool_name)}</code>"
            f"({_esc(step.args)}) &rarr; {_esc(step.result)}</div>"
        )
    if isinstance(step, AnswerStep):
        return f"<div class='step step-answer'>{idx} answer: {_esc(step.text)}</div>"
    return f"<div class='step'>[{index}] {_esc(step)}</div>"


def _render_trace(tr: TrajectoryResult) -> str:
    if tr.error is not None:
        return f"<div class='trace-error'>Error: {_esc(tr.error)}</div>"
    if tr.trajectory is None:
        return "<div class='trace-error'>No trajectory recorded.</div>"

    steps_html = "".join(_render_step(s, i) for i, s in enumerate(tr.trajectory.steps))
    scores_html = "".join(
        f"<span class='score-chip'>{_esc(name)}: {_fmt(r.value)}</span>"
        for name, r in tr.metric_results.items()
    )
    metric_errors_html = "".join(
        f"<div class='trace-error'>{_esc(name)} failed to score: {_esc(err)}</div>"
        for name, err in tr.metric_errors.items()
    )
    tags_html = " ".join(f"<span class=tag>{_esc(t)}</span>" for t in tr.tags)
    final_answer = _esc(tr.trajectory.final_answer)
    return (
        f"<details class='trajectory'><summary>{_esc(tr.question)} "
        f"<span class='tags'>{tags_html}</span>"
        f"<div class='scores'>{scores_html}</div></summary>"
        f"<div class='steps'>{steps_html}</div>"
        f"{metric_errors_html}"
        f"<div class='final-answer'><strong>Final answer:</strong> {final_answer}</div>"
        "</details>"
    )


_CSS = """
body {
  font-family: -apple-system, Segoe UI, sans-serif;
  max-width: 1100px;
  margin: 2rem auto;
  padding: 0 1rem;
  color: #1a1a1a;
}
h1, h2, h3 { border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }
table { border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }
th, td {
  border: 1px solid #ddd;
  padding: 0.4rem 0.6rem;
  text-align: left;
  font-size: 0.9rem;
}
th { background: #f5f5f5; }
.badge {
  display: inline-block;
  padding: 0.1rem 0.5rem;
  border-radius: 0.3rem;
  font-size: 0.75rem;
  margin-left: 0.4rem;
}
.badge-warn { background: #fff3cd; color: #856404; }
.badge-ok { background: #d4edda; color: #155724; }
.trajectory {
  border: 1px solid #ddd;
  border-radius: 0.4rem;
  margin-bottom: 0.6rem;
  padding: 0.5rem 0.8rem;
}
.trajectory summary { cursor: pointer; font-weight: 600; }
.tag {
  background: #eee;
  border-radius: 0.3rem;
  padding: 0.05rem 0.4rem;
  font-size: 0.7rem;
  margin-left: 0.3rem;
}
.scores { font-size: 0.8rem; color: #555; margin-top: 0.3rem; }
.score-chip { display: inline-block; margin-right: 0.6rem; }
.step {
  font-family: ui-monospace, Consolas, monospace;
  font-size: 0.85rem;
  padding: 0.25rem 0;
  border-left: 3px solid #ccc;
  padding-left: 0.6rem;
  margin: 0.3rem 0;
}
.step-retrieval { border-left-color: #4a90d9; }
.step-answer { border-left-color: #2e7d32; }
.step-tool { border-left-color: #b26a00; }
.step-idx { color: #888; margin-right: 0.4rem; }
.trace-error { color: #b00020; font-family: monospace; }
.final-answer { margin-top: 0.5rem; padding-top: 0.4rem; border-top: 1px dashed #ccc; }
"""


def render_report(
    run_result: RunResult, *, cost_summary: dict[str, dict[str, Any]] | None = None
) -> str:
    """Render `run_result` as a single self-contained HTML string."""
    worst = _worst_trajectories(run_result)
    trace_html = "".join(_render_trace(tr) for tr in worst)

    errored = [tr for tr in run_result.trajectory_results if tr.error is not None]
    errors_html = "".join(_render_trace(tr) for tr in errored)
    errors_section = (
        f"<h2>Errored trajectories ({len(errored)})</h2>\n{errors_html}\n" if errored else ""
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>trajeval report — {_esc(run_result.metadata.run_id)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>trajeval report</h1>
<p>
  run <code>{_esc(run_result.metadata.run_id)}</code> ·
  adapter <code>{_esc(run_result.metadata.adapter_name)}</code> ·
  {_esc(run_result.metadata.num_trajectories)} trajectories
  ({_esc(run_result.metadata.num_errors)} errors) ·
  git <code>{_esc(run_result.metadata.git_sha or "n/a")}</code>
</p>

<h2>Aggregate scores</h2>
{_render_aggregate_table(run_result)}

<h2>Per-tag breakdown</h2>
{_render_tag_breakdown(run_result)}

<h2>Cost summary</h2>
{_render_cost_summary(cost_summary)}

{errors_section}
<h2>Worst trajectories</h2>
{trace_html or "<p><em>No trajectories to show.</em></p>"}

</body>
</html>
"""
