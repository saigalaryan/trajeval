"""The output schema `trajeval.runner` writes: one JSON file, no database.

A `RunResult` is a complete, self-describing record of one evaluation run:
what was run against what, every trajectory's full step-by-step record and
per-metric scores, the dataset-level aggregates, and enough metadata (git
SHA, config hash, timing) to make two runs comparable later. The CLI's
`compare`/`report` commands and the web viewer only ever read this one file
— nothing here talks to a database, and the full `Trajectory` is stored
inline specifically so a report can render a trace view without a second
data source to go find.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, JsonValue

from trajeval.metrics.base import MetricResult
from trajeval.types import SCHEMA_VERSION, Trajectory, TrajevalModel


class TrajectoryResult(TrajevalModel):
    """One golden record's outcome: either the trajectory plus every
    metric's score, or why it failed. Never both — see the module-level
    note on error handling.
    """

    golden_id: str
    question: str
    # Copied from the golden record at run time, so a report can slice
    # aggregates by tag from this one file — no need to re-load the
    # original dataset just to know which records were "hard" or "spanish".
    tags: list[str] = Field(default_factory=list)
    # None when the adapter raised before producing a trajectory.
    trajectory_id: str | None = None
    # The full trajectory, stored inline so a report's trace view has
    # everything it needs from this one file — which query failed, what
    # came back, where the agent went wrong. None exactly when trajectory_id
    # is None (the adapter never produced one).
    trajectory: Trajectory | None = None
    metric_results: dict[str, MetricResult] = Field(default_factory=dict)
    # Populated instead of trajectory/metric_results when adapter.run()
    # raised. A failed record is excluded from every metric's aggregate — it
    # was never scored, which is different from having scored zero.
    error: str | None = None
    # metric name -> error message, for metrics whose *scoring* raised on an
    # otherwise-successful trajectory (most commonly a judged metric hitting
    # an unparseable model response). Distinct from `error`: the trajectory
    # itself is fine and every *other* metric's result is still present —
    # only this one metric is missing for this one trajectory, and it's
    # excluded from just that metric's aggregate rather than aborting the
    # whole run.
    metric_errors: dict[str, str] = Field(default_factory=dict)


class RunMetadata(TrajevalModel):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    started_at: datetime
    finished_at: datetime
    # None when not run inside a git checkout (e.g. a packaged install).
    git_sha: str | None = None
    # Hash of what produced this run (currently: dataset content + metric
    # names + adapter name). Once Phase 3's trajeval.yaml exists, this should
    # hash the resolved config instead — tracked, not a final design.
    config_hash: str
    adapter_name: str
    dataset_path: str | None = None
    num_trajectories: int
    num_errors: int
    metric_names: list[str]


class CalibrationState(TrajevalModel):
    """Judge-vs-human agreement for one judged metric on this run.

    A judged metric with no `CalibrationState` entry — or one whose
    `is_calibrated` is False — has never been shown to agree with a human on
    this codebase's questions. Every consumer of `RunResult` (the CLI
    report, the web viewer) must surface that prominently rather than
    letting an uncalibrated score render as a clean, trustworthy number.
    """

    # False until at least MIN_LABELS_FOR_CALIBRATION human labels exist for
    # this metric — see trajeval.calibration.kappa.
    is_calibrated: bool = False
    kappa: float | None = None
    n_labels: int = 0
    # Cohen's kappa sliced by golden-record tag — where judges quietly fall
    # apart most often (e.g. a language slice with much lower agreement than
    # the overall score suggests).
    kappa_by_tag: dict[str, float] = Field(default_factory=dict)


class RunResult(TrajevalModel):
    schema_version: int = SCHEMA_VERSION
    metadata: RunMetadata
    trajectory_results: list[TrajectoryResult]
    # metric name -> that metric's aggregate() output, computed only over
    # trajectories that didn't error.
    aggregate_scores: dict[str, dict[str, JsonValue]]
    # metric name -> calibration state, for every *judged* metric in this
    # run (deterministic metrics never appear here — they need no judge
    # agreement check). Absent entirely only for a run with no judged
    # metrics at all.
    calibration: dict[str, CalibrationState] = Field(default_factory=dict)
