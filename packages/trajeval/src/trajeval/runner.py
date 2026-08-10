"""trajeval.runner: run an adapter over a golden dataset and score it.

Takes plain Python objects in (an `AgentAdapter`, a list of `GoldenRecord`,
a list of `Metric`) and writes a `RunResult` out. Knows nothing about YAML
config or the CLI — `trajeval.cli` builds on top of this in Phase 3.

`adapter.run()` is a blocking call by contract (see `AgentAdapter`), so
concurrency here is a thread pool, not asyncio — we don't control what's
inside it, and a thread pool is the right tool for "call this synchronous,
probably-I/O-bound function N times at once."
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from trajeval.adapters import AgentAdapter
from trajeval.calibration.kappa import compute_calibration
from trajeval.calibration.labels import JUDGED_METRIC_NAMES, HumanLabel, load_labels
from trajeval.metrics.base import Metric, MetricResult
from trajeval.results import RunMetadata, RunResult, TrajectoryResult
from trajeval.types import GoldenRecord, Trajectory


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return proc.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _config_hash(
    golden_records: Sequence[GoldenRecord], metric_names: Sequence[str], adapter_name: str
) -> str:
    """A stable fingerprint of "what produced this run".

    Two RunResults with different hashes were produced by a meaningfully
    different setup (different dataset content, different metrics, or a
    different adapter) and shouldn't be diffed by `trajeval compare` without
    a warning. See RunMetadata.config_hash for the caveat about this being
    an interim definition ahead of Phase 3's config file.
    """
    payload = {
        "golden_ids": sorted(g.id for g in golden_records),
        "metric_names": sorted(metric_names),
        "adapter_name": adapter_name,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def run(
    adapter: AgentAdapter,
    golden_records: Sequence[GoldenRecord],
    metrics: Sequence[Metric],
    *,
    concurrency: int = 4,
    adapter_name: str | None = None,
    dataset_path: str | None = None,
    labels_path: str | Path | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> RunResult:
    """Run `adapter` over every golden record, score with every metric.

    A record whose `adapter.run()` call raises is recorded with its error
    message and excluded from scoring and from every metric's aggregate,
    rather than aborting the whole run — see RunMetadata.num_errors.

    A record whose *trajectory* is produced fine but whose scoring raises
    for one particular metric (most commonly a judged metric hitting an
    unparseable model response) is handled the same way, one level down:
    that one metric is excluded from its own aggregate for that trajectory
    — see TrajectoryResult.metric_errors — while every other metric's score
    on that trajectory, and every other trajectory entirely, is unaffected.

    Every judged metric present in `metrics` (see
    `trajeval.calibration.labels.JUDGED_METRIC_NAMES`) gets a
    `CalibrationState` entry in the result, whether or not `labels_path` is
    given — with no labels, or too few, it's simply `is_calibrated=False`.
    This is deliberate: a judged metric's `RunResult` entry must never be
    silently missing a calibration state.

    `on_progress(completed, total)`, if given, is called once per golden
    record as it finishes (adapter call + all metric scoring), in
    completion order — which, under a thread pool, is not dataset order.
    This module deliberately doesn't depend on any particular progress-bar
    library; `trajeval.cli` wires this to `rich` itself.
    """
    ids = [g.id for g in golden_records]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"golden_records contains duplicate ids: {dupes}")

    started_at = datetime.now(UTC)
    resolved_adapter_name = adapter_name or type(adapter).__name__

    errors: dict[str, str] = {}
    trajectories_by_golden: dict[str, Trajectory] = {}
    metric_results_by_golden: dict[str, dict[str, MetricResult]] = {}
    metric_errors_by_golden: dict[str, dict[str, str]] = {}
    per_metric_results: dict[str, list[MetricResult]] = {m.name: [] for m in metrics}

    total = len(golden_records)
    completed = 0

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        future_to_golden = {pool.submit(adapter.run, g.question): g for g in golden_records}
        for future in as_completed(future_to_golden):
            golden = future_to_golden[future]
            try:
                trajectory: Trajectory | None = future.result()
            except Exception as exc:  # noqa: BLE001 - deliberately broad: any adapter failure is recorded, not fatal
                errors[golden.id] = f"{type(exc).__name__}: {exc}"
                trajectory = None

            if trajectory is not None:
                trajectory.golden_id = trajectory.golden_id or golden.id
                trajectories_by_golden[golden.id] = trajectory
                scores: dict[str, MetricResult] = {}
                metric_errors: dict[str, str] = {}
                for metric in metrics:
                    try:
                        result = metric.score(trajectory, golden)
                    except Exception as exc:  # noqa: BLE001 - a judge/parsing failure on one metric
                        # must not take down every other metric's result for this
                        # trajectory, let alone the whole run — see TrajectoryResult.metric_errors.
                        metric_errors[metric.name] = f"{type(exc).__name__}: {exc}"
                        continue
                    scores[metric.name] = result
                    per_metric_results[metric.name].append(result)
                metric_results_by_golden[golden.id] = scores
                if metric_errors:
                    metric_errors_by_golden[golden.id] = metric_errors

            completed += 1
            if on_progress is not None:
                on_progress(completed, total)

    trajectory_results = []
    for golden in golden_records:
        stored_trajectory = trajectories_by_golden.get(golden.id)
        trajectory_results.append(
            TrajectoryResult(
                golden_id=golden.id,
                question=golden.question,
                tags=golden.tags,
                trajectory_id=stored_trajectory.id if stored_trajectory is not None else None,
                trajectory=stored_trajectory,
                metric_results=metric_results_by_golden.get(golden.id, {}),
                error=errors.get(golden.id),
                metric_errors=metric_errors_by_golden.get(golden.id, {}),
            )
        )

    aggregate_scores = {
        metric.name: metric.aggregate(per_metric_results[metric.name]) for metric in metrics
    }

    finished_at = datetime.now(UTC)
    metadata = RunMetadata(
        run_id=uuid4().hex,
        started_at=started_at,
        finished_at=finished_at,
        git_sha=_git_sha(),
        config_hash=_config_hash(golden_records, [m.name for m in metrics], resolved_adapter_name),
        adapter_name=resolved_adapter_name,
        dataset_path=dataset_path,
        num_trajectories=len(golden_records),
        num_errors=len(errors),
        metric_names=[m.name for m in metrics],
    )

    run_result = RunResult(
        metadata=metadata,
        trajectory_results=trajectory_results,
        aggregate_scores=aggregate_scores,
    )

    judged_metric_names = [m.name for m in metrics if m.name in JUDGED_METRIC_NAMES]
    if judged_metric_names:
        labels: list[HumanLabel] = load_labels(labels_path) if labels_path is not None else []
        run_result.calibration = {
            metric_name: compute_calibration(run_result, labels, metric_name)
            for metric_name in judged_metric_names
        }

    return run_result


def recalibrate(run_result: RunResult, labels_path: str | Path) -> RunResult:
    """Recompute calibration state for a *previously saved* `RunResult`
    against a labels file, and return the updated copy.

    This — not `run(..., labels_path=...)` — is the realistic calibration
    workflow: `trajeval label` walks a human through trajectories from one
    specific saved run, so the labels it produces are only joinable against
    *that same run's* trajectory ids. Re-invoking `run()` calls the adapter
    again and gets fresh `uuid4()` trajectory ids that won't match any
    existing label — `run(..., labels_path=...)` only helps when the
    adapter itself assigns stable, reproducible trajectory ids.

    Deterministic and judged metric names both pass through this
    unmodified; only entries for judged metric names present in
    `run_result.metadata.metric_names` are (re)computed.
    """
    judged_metric_names = [
        name for name in run_result.metadata.metric_names if name in JUDGED_METRIC_NAMES
    ]
    if not judged_metric_names:
        return run_result

    labels = load_labels(labels_path)
    updated = run_result.model_copy(deep=True)
    updated.calibration = {
        metric_name: compute_calibration(run_result, labels, metric_name)
        for metric_name in judged_metric_names
    }
    return updated


def save_run_result(result: RunResult, path: str | Path) -> None:
    """Write a RunResult to disk as pretty-printed JSON."""
    Path(path).write_text(result.model_dump_json(indent=2), encoding="utf-8")


def load_run_result(path: str | Path) -> RunResult:
    return RunResult.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_golden_dataset(path: str | Path) -> list[GoldenRecord]:
    """Load a JSONL golden dataset: one GoldenRecord per non-blank line."""
    records: list[GoldenRecord] = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(GoldenRecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid GoldenRecord JSON: {exc}") from exc
    return records
