"""Smoke tests for trajeval.cli — one per command, exercising the real
Typer app end-to-end against a temp directory. Deeper logic (config
parsing, regression checks, report rendering, labeling) is already unit
tested at the library level; these just confirm the CLI wiring works.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from trajeval.cli import _calibration_badge, _judge_cost_line, app
from trajeval.cost import CostTracker
from trajeval.results import CalibrationState

runner = CliRunner()


def _write_dataset(path: Path) -> None:
    path.write_text(
        '{"id": "g1", "question": "What is 2+2?", "reference_answer": "4", '
        '"retrieval_required": false, "min_steps": 1}\n',
        encoding="utf-8",
    )


def _write_adapter_module(tmp_path: Path) -> None:
    (tmp_path / "my_test_agent.py").write_text(
        "from trajeval import CallableAdapter\n\n"
        "def _answer(question):\n"
        '    return {"final_answer": "4", "steps": [{"step_type": "answer", "text": "4"}]}\n\n'
        "MyAdapter = CallableAdapter(_answer)\n",
        encoding="utf-8",
    )


def test_calibration_badge_non_judged_metric_is_blank() -> None:
    assert _calibration_badge("retrieval_necessity", None) == ""


def test_calibration_badge_judged_metric_no_calibration_state_is_uncalibrated() -> None:
    assert _calibration_badge("query_quality", None) == " [UNCALIBRATED]"


def test_calibration_badge_judged_metric_not_calibrated() -> None:
    cal = CalibrationState(is_calibrated=False, kappa=1.0, n_labels=2, kappa_by_tag={})
    assert _calibration_badge("query_quality", cal) == " [UNCALIBRATED]"


def test_calibration_badge_judged_metric_calibrated_shows_kappa() -> None:
    cal = CalibrationState(is_calibrated=True, kappa=0.756, n_labels=50, kappa_by_tag={})
    assert _calibration_badge("query_quality", cal) == " [κ=0.76]"


def test_judge_cost_line_degenerate_no_judge_calls_is_none() -> None:
    assert _judge_cost_line(CostTracker()) is None


def test_judge_cost_line_reports_total_when_costs_recorded() -> None:
    tracker = CostTracker()
    tracker.record("query_quality", "claude-opus-5", 100, 50)
    assert _judge_cost_line(tracker) == f"Judge cost: ${tracker.total_cost_usd():.4f}"


def test_judge_cost_line_unknown_model_cost_is_none() -> None:
    tracker = CostTracker()
    tracker.record("query_quality", "some-unpriced-model", 100, 50)
    assert _judge_cost_line(tracker) is None


def test_init_scaffolds_config_and_adapter(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "trajeval.yaml").exists()
    assert (tmp_path / "my_agent.py").exists()


def test_init_skips_existing_files(tmp_path: Path) -> None:
    (tmp_path / "trajeval.yaml").write_text("existing content", encoding="utf-8")
    result = runner.invoke(app, ["init", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "trajeval.yaml").read_text(encoding="utf-8") == "existing content"


def test_init_without_seed_dataset_flag_does_not_write_one(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert not (tmp_path / "datasets" / "seed" / "seed.jsonl").exists()


def test_init_with_seed_dataset_writes_a_usable_dataset(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "--dir", str(tmp_path), "--with-seed-dataset"])
    assert result.exit_code == 0
    dataset_path = tmp_path / "datasets" / "seed" / "seed.jsonl"
    assert dataset_path.exists()

    # Round-trips through validate_dataset cleanly — proves it's not just a
    # file that exists, but one trajeval run/validate can actually use.
    from trajeval.dataset_validation import validate_dataset

    report = validate_dataset(dataset_path)
    assert report.ok
    assert report.num_records > 0


def test_init_with_seed_dataset_skips_existing_file(tmp_path: Path) -> None:
    dataset_path = tmp_path / "datasets" / "seed" / "seed.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("existing content", encoding="utf-8")

    result = runner.invoke(app, ["init", "--dir", str(tmp_path), "--with-seed-dataset"])

    assert result.exit_code == 0
    assert dataset_path.read_text(encoding="utf-8") == "existing content"


def test_validate_end_to_end_clean_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)

    result = runner.invoke(app, ["validate", str(dataset)])

    assert result.exit_code == 0, result.output
    assert "1 record(s), no issues found" in result.output


def test_validate_end_to_end_reports_issues_and_exits_nonzero(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"id": "g1", "question": "q", "reference_answer": "a", '
        '"retrieval_required": false, "min_steps": 1}\n'
        "not json at all\n"
        '{"id": "g1", "question": "dup", "reference_answer": "a", '
        '"retrieval_required": false, "min_steps": 1}\n',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["validate", str(dataset)])

    assert result.exit_code == 1
    assert "line 2" in result.output
    assert "duplicate id" in result.output


def test_run_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_adapter_module(tmp_path)
    _write_dataset(tmp_path / "dataset.jsonl")
    (tmp_path / "trajeval.yaml").write_text(
        "adapter: my_test_agent:MyAdapter\n"
        "dataset: dataset.jsonl\n"
        "metrics:\n  - name: retrieval_necessity\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["run", "--config", "trajeval.yaml", "--out", "results/out.json"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "results" / "out.json").exists()
    assert "retrieval_necessity" in result.output


def test_report_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_adapter_module(tmp_path)
    _write_dataset(tmp_path / "dataset.jsonl")
    (tmp_path / "trajeval.yaml").write_text(
        "adapter: my_test_agent:MyAdapter\n"
        "dataset: dataset.jsonl\n"
        "metrics:\n  - name: retrieval_necessity\n",
        encoding="utf-8",
    )
    runner.invoke(app, ["run", "--config", "trajeval.yaml", "--out", "results/out.json"])

    result = runner.invoke(app, ["report", "results/out.json", "--html", "report.html"])
    assert result.exit_code == 0, result.output
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "trajeval report" in html


def test_compare_end_to_end_no_regression(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_adapter_module(tmp_path)
    _write_dataset(tmp_path / "dataset.jsonl")
    (tmp_path / "trajeval.yaml").write_text(
        "adapter: my_test_agent:MyAdapter\n"
        "dataset: dataset.jsonl\n"
        "metrics:\n  - name: retrieval_necessity\n",
        encoding="utf-8",
    )
    runner.invoke(app, ["run", "--config", "trajeval.yaml", "--out", "baseline.json"])
    runner.invoke(app, ["run", "--config", "trajeval.yaml", "--out", "candidate.json"])

    result = runner.invoke(app, ["compare", "baseline.json", "candidate.json"])
    assert result.exit_code == 0, result.output
    assert "retrieval_necessity" in result.output


def test_compare_end_to_end_regression_fails_build(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_dataset(tmp_path / "dataset.jsonl")
    # Two different adapters against the same retrieval_required=false
    # golden: one answers directly (necessity_score=1.0, correct_skip), the
    # other always retrieves (necessity_score=0.0, over_retrieval) — a real
    # regression between baseline and candidate, not just a diff.
    (tmp_path / "good_agent.py").write_text(
        "from trajeval import CallableAdapter\n\n"
        "def _answer(question):\n"
        '    return {"final_answer": "4", "steps": [{"step_type": "answer", "text": "4"}]}\n\n'
        "GoodAdapter = CallableAdapter(_answer)\n",
        encoding="utf-8",
    )
    (tmp_path / "bad_agent.py").write_text(
        "from trajeval import CallableAdapter\n\n"
        "def _answer(question):\n"
        "    return {\n"
        '        "final_answer": "4",\n'
        '        "steps": [\n'
        '            {"step_type": "retrieval", "query": question, "chunks": []},\n'
        '            {"step_type": "answer", "text": "4"},\n'
        "        ],\n"
        "    }\n\n"
        "BadAdapter = CallableAdapter(_answer)\n",
        encoding="utf-8",
    )
    (tmp_path / "baseline.yaml").write_text(
        "adapter: good_agent:GoodAdapter\n"
        "dataset: dataset.jsonl\n"
        "metrics:\n  - name: retrieval_necessity\n",
        encoding="utf-8",
    )
    (tmp_path / "candidate.yaml").write_text(
        "adapter: bad_agent:BadAdapter\n"
        "dataset: dataset.jsonl\n"
        "metrics:\n  - name: retrieval_necessity\n"
        "regression_thresholds:\n"
        "  - metric: retrieval_necessity\n"
        "    key: necessity_score\n"
        "    tolerance: 0.05\n",
        encoding="utf-8",
    )
    runner.invoke(app, ["run", "--config", "baseline.yaml", "--out", "baseline.json"])
    runner.invoke(app, ["run", "--config", "candidate.yaml", "--out", "candidate.json"])

    result = runner.invoke(
        app, ["compare", "baseline.json", "candidate.json", "--config", "candidate.yaml"]
    )

    assert result.exit_code == 1
    assert "regressed" in result.output


def test_label_end_to_end_records_scripted_answers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_adapter_module(tmp_path)
    _write_dataset(tmp_path / "dataset.jsonl")
    (tmp_path / "trajeval.yaml").write_text(
        "adapter: my_test_agent:MyAdapter\n"
        "dataset: dataset.jsonl\n"
        "metrics:\n  - name: retrieval_necessity\n",
        encoding="utf-8",
    )
    runner.invoke(app, ["run", "--config", "trajeval.yaml", "--out", "results/out.json"])

    # retrieval_necessity isn't a judged metric, so there's nothing to label
    # — this exercises the full label_command wiring (load run, load
    # dataset, call run_labeling_session, report the count) via its
    # legitimate "0 candidates" path rather than needing a real judge run.
    result = runner.invoke(
        app,
        [
            "label",
            "--run",
            "results/out.json",
            "--dataset",
            "dataset.jsonl",
            "--metric",
            "recovery",
            "--labels",
            "labels.jsonl",
            "--labeler",
            "tester",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Recorded 0 label(s)" in result.output


def test_serve_command_wires_args_through(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        "trajeval.cli.serve_viewer",
        lambda results_dir, *, file, port, open_browser: calls.append(
            {"results_dir": results_dir, "file": file, "port": port, "open_browser": open_browser}
        ),
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    result = runner.invoke(
        app, ["serve", str(results_dir), "--file", "latest.json", "--port", "9000", "--no-browser"]
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0] == {
        "results_dir": results_dir,
        "file": "latest.json",
        "port": 9000,
        "open_browser": False,
    }


def test_serve_command_reports_viewer_not_bundled(tmp_path: Path, monkeypatch) -> None:
    from trajeval.serve import ViewerNotBundledError

    def _raise(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise ViewerNotBundledError("no viewer bundled")

    monkeypatch.setattr("trajeval.cli.serve_viewer", _raise)
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    result = runner.invoke(app, ["serve", str(results_dir)])

    assert result.exit_code == 1
    assert "no viewer bundled" in result.output


def test_compare_degenerate_missing_baseline_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["compare", "does-not-exist.json", "also-missing.json"])
    assert result.exit_code != 0


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "run" in result.output
    assert "compare" in result.output
    assert "label" in result.output
    assert "report" in result.output
    assert "init" in result.output


def test_version_flag() -> None:
    from trajeval import __version__

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output
