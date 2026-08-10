"""The core of `trajeval label`: walks a human through trajectories one at a
time for a given judged metric, recording their judgment against the same
category set the metric's own judge verdict uses (see
`trajeval.calibration.verdicts`).

Deliberately does **not** show the judge's own verdict while labeling — that
would anchor the human toward agreeing with it, defeating the point of an
independent comparison. It's split from the interactive I/O
(`input_fn`/`print_fn` are injectable) so the walking/filtering/recording
logic can be tested without a terminal attached.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from trajeval.calibration.labels import HumanLabel, append_label, load_labels
from trajeval.calibration.verdicts import judge_verdict
from trajeval.results import RunResult
from trajeval.types import (
    AnswerStep,
    GoldenRecord,
    RetrievalStep,
    ThoughtStep,
    ToolStep,
    Trajectory,
)

VERDICT_OPTIONS: dict[str, list[str]] = {
    "recovery": ["answered_from_bad_context", "correctly_abstained"],
    "query_quality": ["1", "2", "3", "4", "5"],
    "faithfulness": ["fully_supported", "not_fully_supported"],
}


def format_trajectory_for_labeling(trajectory: Trajectory, golden: GoldenRecord) -> str:
    """Render a trajectory's context for a human labeler. Never includes the
    judge's verdict — see the module docstring."""
    lines = [f"Question: {golden.question}", ""]
    for i, step in enumerate(trajectory.steps):
        if isinstance(step, ThoughtStep):
            lines.append(f"  [{i}] thought: {step.text}")
        elif isinstance(step, RetrievalStep):
            lines.append(f"  [{i}] retrieval: query={step.query!r}")
            for chunk in step.chunks:
                snippet = chunk.text if len(chunk.text) <= 200 else chunk.text[:200] + "..."
                lines.append(f"        - [{chunk.doc_id}] {snippet}")
        elif isinstance(step, ToolStep):
            lines.append(f"  [{i}] tool: {step.tool_name}({step.args}) -> {step.result}")
        elif isinstance(step, AnswerStep):
            lines.append(f"  [{i}] answer: {step.text}")
    lines.append("")
    lines.append(f"Final answer: {trajectory.final_answer}")
    return "\n".join(lines)


def run_labeling_session(
    run_result: RunResult,
    trajectories: dict[str, Trajectory],
    goldens: dict[str, GoldenRecord],
    metric_name: str,
    n: int,
    labeler: str,
    labels_path: str | Path,
    *,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> int:
    """Label up to `n` trajectories that have a judge verdict for
    `metric_name` and no existing label from `labeler`. Returns the number
    of labels recorded.
    """
    options = VERDICT_OPTIONS.get(metric_name)
    if options is None:
        raise ValueError(
            f"{metric_name!r} is not a judged metric with a labeling scheme. "
            f"Known metrics: {sorted(VERDICT_OPTIONS)}"
        )

    already_labeled = {
        (lbl.trajectory_id, lbl.metric_name, lbl.labeler) for lbl in load_labels(labels_path)
    }

    candidates = []
    for tr in run_result.trajectory_results:
        if tr.trajectory_id is None:
            continue
        metric_result = tr.metric_results.get(metric_name)
        if metric_result is None or judge_verdict(metric_name, metric_result) is None:
            continue
        if (tr.trajectory_id, metric_name, labeler) in already_labeled:
            continue
        candidates.append(tr)

    recorded = 0
    for tr in candidates[:n]:
        assert tr.trajectory_id is not None
        trajectory = trajectories.get(tr.trajectory_id)
        golden = goldens.get(tr.golden_id)
        if trajectory is None or golden is None:
            continue

        print_fn("=" * 70)
        print_fn(format_trajectory_for_labeling(trajectory, golden))
        print_fn("")
        print_fn(f"Your judgment for '{metric_name}' — options: {', '.join(options)}")

        raw = input_fn("> ").strip()
        while raw not in options:
            print_fn(f"Not a valid option. Choose one of: {', '.join(options)}")
            raw = input_fn("> ").strip()

        label = HumanLabel(
            trajectory_id=tr.trajectory_id,
            golden_id=tr.golden_id,
            metric_name=metric_name,
            human_verdict=raw,
            labeler=labeler,
            tags=golden.tags,
        )
        append_label(labels_path, label)
        recorded += 1

    return recorded
