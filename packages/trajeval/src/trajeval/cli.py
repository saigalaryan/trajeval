"""trajeval's command-line interface — a thin shell over the library. Every
command here is a few lines gluing config/runner/report/calibration
together; no logic lives here that isn't already tested at the library
level (see trajeval.config, trajeval.compare, trajeval.report,
trajeval.calibration).
"""

from __future__ import annotations

import getpass
from pathlib import Path

import typer
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from trajeval import __version__
from trajeval import config as config_module
from trajeval import runner as runner_module
from trajeval.calibration.cli import run_labeling_session
from trajeval.calibration.labels import JUDGED_METRIC_NAMES
from trajeval.compare import check_regressions, diff_aggregates, format_comparison_table
from trajeval.cost import CostTracker
from trajeval.dataset_validation import validate_dataset
from trajeval.report import render_report
from trajeval.results import CalibrationState
from trajeval.serve import ViewerNotBundledError
from trajeval.serve import serve as serve_viewer

app = typer.Typer(
    name="trajeval",
    help="Evaluation harness for agentic RAG systems: scores the trajectory, not just the answer.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"trajeval {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show trajeval's version and exit.",
    ),
) -> None:
    pass


def _calibration_badge(metric_name: str, cal: CalibrationState | None) -> str:
    """Pulled out of run_command as its own function so the calibrated /
    uncalibrated / not-a-judged-metric branches are unit-testable directly,
    without needing a real (or heavily faked) judged run to reach 50+
    labels' worth of a genuinely `is_calibrated=True` state."""
    if metric_name not in JUDGED_METRIC_NAMES:
        return ""
    if cal is None or not cal.is_calibrated:
        return " [UNCALIBRATED]"
    return f" [κ={cal.kappa:.2f}]"


def _judge_cost_line(tracker: CostTracker) -> str | None:
    """None means "nothing judged happened this run" (no judged metrics
    configured, or every judge call was a cache hit) — distinct from "$0.00
    of known cost", which would still print."""
    total_cost = tracker.total_cost_usd()
    if total_cost is None or not tracker.summary():
        return None
    return f"Judge cost: ${total_cost:.4f}"


@app.command("validate")
def validate_command(
    dataset: Path = typer.Argument(..., exists=True, help="Golden dataset JSONL to lint"),
) -> None:
    """Lint a golden dataset before running anything against it.

    Catches malformed JSON, missing/invalid fields, and duplicate ids in
    under a second — everything `trajeval run` would eventually reject one
    line at a time, reported all at once instead of discovered partway
    through a slow, and for judged metrics costly, real run.
    """
    report = validate_dataset(dataset)

    if report.ok:
        typer.echo(f"{dataset}: {report.num_records} record(s), no issues found.")
        return

    for issue in report.issues:
        where = f"line {issue.line}" if issue.line is not None else "dataset"
        typer.echo(f"  {where}: {issue.message}")
    typer.echo(f"\n{dataset}: {report.num_records} valid record(s), {len(report.issues)} issue(s).")
    raise typer.Exit(code=1)


@app.command("run")
def run_command(
    config: Path = typer.Option(..., "--config", exists=True, help="Path to trajeval.yaml"),
    out: Path = typer.Option(
        Path("results/latest.json"), "--out", help="Output RunResult JSON path"
    ),
) -> None:
    """Run the configured adapter against the configured dataset and score it."""
    cfg = config_module.load_config(config)
    goldens = runner_module.load_golden_dataset(cfg.dataset)
    adapter = config_module.resolve_adapter(cfg.adapter)
    tracker = CostTracker()
    metrics = config_module.build_metrics(cfg, cost_tracker=tracker)

    # Trajectories finish in completion order under the thread pool, not
    # dataset order — a plain counter is all `on_progress` gives us, which
    # is exactly what a progress bar needs and nothing more.
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Scoring trajectories", total=len(goldens))

        def on_progress(completed: int, total: int) -> None:
            progress.update(task, completed=completed)

        result = runner_module.run(
            adapter,
            goldens,
            metrics,
            concurrency=cfg.concurrency,
            dataset_path=cfg.dataset,
            labels_path=cfg.labels_path,
            on_progress=on_progress,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    runner_module.save_run_result(result, out)

    total_metric_errors = sum(len(tr.metric_errors) for tr in result.trajectory_results)

    typer.echo(f"Wrote {out}")
    typer.echo(
        f"  {result.metadata.num_trajectories} trajectories, {result.metadata.num_errors} errors"
        + (f", {total_metric_errors} metric-scoring failure(s)" if total_metric_errors else "")
    )
    for metric_name, agg in result.aggregate_scores.items():
        cal = result.calibration.get(metric_name)
        typer.echo(f"  {metric_name}{_calibration_badge(metric_name, cal)}: {agg}")

    cost_line = _judge_cost_line(tracker)
    if cost_line is not None:
        typer.echo(cost_line)


@app.command("compare")
def compare_command(
    baseline: Path = typer.Argument(..., exists=True, help="Baseline RunResult JSON"),
    candidate: Path = typer.Argument(..., exists=True, help="Candidate RunResult JSON"),
    config: Path | None = typer.Option(
        None, "--config", help="trajeval.yaml providing regression_thresholds"
    ),
    fail_on_regression: bool = typer.Option(
        True, help="Exit non-zero if any configured threshold regressed"
    ),
) -> None:
    """Diff two RunResults and (optionally) fail on configured regressions.
    Handles a missing baseline gracefully: if `baseline` doesn't exist,
    that's a caller error caught by Typer's `exists=True` — for a genuinely
    *first* run with nothing to compare against, don't call this command.
    """
    baseline_result = runner_module.load_run_result(baseline)
    candidate_result = runner_module.load_run_result(candidate)

    thresholds = config_module.load_config(config).regression_thresholds if config else []
    deltas = diff_aggregates(baseline_result, candidate_result)
    checked = check_regressions(deltas, thresholds)

    typer.echo(format_comparison_table(checked))

    regressions = [d for d in checked if d.regressed]
    if regressions:
        typer.echo(f"\n{len(regressions)} metric(s) regressed beyond their configured threshold.")
        if fail_on_regression:
            raise typer.Exit(code=1)


@app.command("label")
def label_command(
    run_result: Path = typer.Option(
        ..., "--run", exists=True, help="RunResult JSON to label trajectories from"
    ),
    dataset: Path = typer.Option(
        ..., "--dataset", exists=True, help="Golden dataset JSONL (for tags)"
    ),
    metric: str = typer.Option(
        ..., "--metric", help="Judged metric to label: query_quality | recovery | faithfulness"
    ),
    n: int = typer.Option(50, "--n", help="Max number of trajectories to label this session"),
    labels: Path = typer.Option(Path("labels.jsonl"), "--labels", help="Label file to append to"),
    labeler: str = typer.Option(
        None, "--labeler", help="Your name/id (defaults to the OS username)"
    ),
) -> None:
    """Walk through trajectories one at a time, recording your judgment for
    calibration. See trajeval.calibration for how this feeds Cohen's kappa.
    """
    result = runner_module.load_run_result(run_result)
    goldens = {g.id: g for g in runner_module.load_golden_dataset(dataset)}
    trajectories = {
        tr.trajectory_id: tr.trajectory
        for tr in result.trajectory_results
        if tr.trajectory_id is not None and tr.trajectory is not None
    }

    recorded = run_labeling_session(
        result,
        trajectories,
        goldens,
        metric,
        n,
        labeler or getpass.getuser(),
        labels,
    )
    typer.echo(f"\nRecorded {recorded} label(s) to {labels}")


@app.command("report")
def report_command(
    run_result: Path = typer.Argument(..., exists=True, help="RunResult JSON to render"),
    html: Path = typer.Option(Path("report.html"), "--html", help="Output HTML path"),
) -> None:
    """Render a self-contained HTML report for one RunResult."""
    result = runner_module.load_run_result(run_result)
    html.parent.mkdir(parents=True, exist_ok=True)
    html.write_text(render_report(result), encoding="utf-8")
    typer.echo(f"Wrote {html}")


@app.command("serve")
def serve_command(
    results_dir: Path = typer.Argument(
        Path("results"), exists=True, file_okay=False, help="Directory of RunResult JSON files"
    ),
    file: str = typer.Option(
        None, "--file", help="Filename inside results_dir to open directly (e.g. latest.json)"
    ),
    port: int = typer.Option(8000, "--port", help="Local port to serve on"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't auto-open a browser tab"),
) -> None:
    """Serve the bundled web viewer locally, pointed at a results directory.
    No Node.js required — the viewer ships prebuilt inside this package.
    Blocks until Ctrl+C.
    """
    try:
        serve_viewer(results_dir, file=file, port=port, open_browser=not no_browser)
    except ViewerNotBundledError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


_EXAMPLE_CONFIG = """\
# trajeval config — see https://github.com/saigalaryan/trajeval for the full reference.
adapter: my_agent:MyAdapter   # module.path:ClassName | :factory_fn | :instance
dataset: datasets/seed/seed.jsonl
judge_provider: anthropic     # anthropic | openai
judge_model: claude-opus-5    # e.g. gpt-4o-mini if judge_provider is openai
concurrency: 4
out_dir: results

metrics:
  - name: retrieval_necessity
  - name: trajectory_efficiency
  - name: termination
  # Judged metrics need a JudgeClient — see trajeval.judge. Uncomment once
  # you've reviewed trajeval.calibration and are ready to hand-label:
  # - name: query_quality
  # - name: recovery
  # - name: faithfulness

regression_thresholds:
  - metric: retrieval_necessity
    key: necessity_score
    tolerance: 0.05
"""

_EXAMPLE_ADAPTER = '''\
"""Example adapter — replace with your own agent."""

from trajeval import CallableAdapter


def _my_agent(question: str) -> dict:
    """Wire this up to your real agent. Must return a trajectory-shaped
    dict: {"final_answer": str, "steps": [...]}."""
    return {
        "final_answer": "TODO: call your agent here",
        "steps": [{"step_type": "answer", "text": "TODO: call your agent here"}],
    }


MyAdapter = CallableAdapter(_my_agent)
'''


_SEED_DATASET = Path(__file__).parent / "_data" / "seed.jsonl"


@app.command("init")
def init_command(
    directory: Path = typer.Option(Path("."), "--dir", help="Directory to scaffold into"),
    with_seed_dataset: bool = typer.Option(
        False,
        "--with-seed-dataset",
        help="Also copy trajeval's bundled example golden dataset to datasets/seed/seed.jsonl",
    ),
) -> None:
    """Scaffold a trajeval.yaml config and an example adapter."""
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "trajeval.yaml"
    adapter_path = directory / "my_agent.py"

    for path, content in ((config_path, _EXAMPLE_CONFIG), (adapter_path, _EXAMPLE_ADAPTER)):
        if path.exists():
            typer.echo(f"Skipped {path} (already exists)")
            continue
        path.write_text(content, encoding="utf-8")
        typer.echo(f"Wrote {path}")

    if with_seed_dataset:
        dataset_path = directory / "datasets" / "seed" / "seed.jsonl"
        if dataset_path.exists():
            typer.echo(f"Skipped {dataset_path} (already exists)")
        else:
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text(_SEED_DATASET.read_text(encoding="utf-8"), encoding="utf-8")
            typer.echo(f"Wrote {dataset_path}")

    typer.echo(
        "\nNext: edit my_agent.py to call your real agent, then run "
        "`trajeval run --config trajeval.yaml`"
    )


def main() -> None:  # pragma: no cover - trivial entrypoint glue, exercised via the console script
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
