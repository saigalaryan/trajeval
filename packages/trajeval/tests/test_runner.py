"""Tests for trajeval.runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from trajeval.adapters import CallableAdapter
from trajeval.calibration.labels import HumanLabel, append_label
from trajeval.judge.client import FakeJudgeClient
from trajeval.metrics.recovery import RecoveryMetric
from trajeval.metrics.retrieval_necessity import RetrievalNecessityMetric
from trajeval.results import RunResult
from trajeval.runner import load_golden_dataset, load_run_result, recalibrate, run, save_run_result
from trajeval.types import GoldenRecord

metric = RetrievalNecessityMetric()


def _golden(id_: str, *, retrieval_required: bool) -> GoldenRecord:
    return GoldenRecord(
        id=id_,
        question=f"question {id_}",
        reference_answer="answer",
        retrieval_required=retrieval_required,
        min_steps=1,
    )


def _always_answers_directly(question: str) -> dict:
    return {"final_answer": "42", "steps": [{"step_type": "answer", "text": "42"}]}


def _always_retrieves(question: str) -> dict:
    return {
        "final_answer": "42",
        "steps": [
            {"step_type": "retrieval", "query": question, "chunks": []},
            {"step_type": "answer", "text": "42"},
        ],
    }


# ---------------------------------------------------------------------------
# run() — happy path
# ---------------------------------------------------------------------------


def test_run_scores_every_golden_record() -> None:
    goldens = [_golden("g1", retrieval_required=False), _golden("g2", retrieval_required=False)]
    adapter = CallableAdapter(_always_answers_directly)

    result = run(adapter, goldens, [metric])

    assert result.metadata.num_trajectories == 2
    assert result.metadata.num_errors == 0
    assert len(result.trajectory_results) == 2
    assert result.aggregate_scores["retrieval_necessity"]["necessity_score"] == 1.0


def test_run_records_over_retrieval_when_agent_always_searches() -> None:
    goldens = [_golden("g1", retrieval_required=False)]
    adapter = CallableAdapter(_always_retrieves)

    result = run(adapter, goldens, [metric])

    tr = result.trajectory_results[0]
    assert tr.metric_results["retrieval_necessity"].details["outcome"] == "over_retrieval"
    assert result.aggregate_scores["retrieval_necessity"]["over_retrieval"] == 1


def test_run_metadata_is_populated() -> None:
    goldens = [_golden("g1", retrieval_required=False)]
    adapter = CallableAdapter(_always_answers_directly)

    result = run(
        adapter, goldens, [metric], adapter_name="my-adapter", dataset_path="datasets/x.jsonl"
    )

    assert result.metadata.adapter_name == "my-adapter"
    assert result.metadata.dataset_path == "datasets/x.jsonl"
    assert result.metadata.metric_names == ["retrieval_necessity"]
    assert result.metadata.started_at <= result.metadata.finished_at
    # not a git repo in this environment; must degrade to None, not raise
    assert result.metadata.git_sha is None


# ---------------------------------------------------------------------------
# run() — failure handling
# ---------------------------------------------------------------------------


def test_run_records_error_and_continues_when_adapter_raises() -> None:
    def flaky(question: str) -> dict:
        if "g2" in question:
            raise RuntimeError("simulated API failure")
        return _always_answers_directly(question)

    goldens = [
        _golden("g1", retrieval_required=False),
        _golden("g2", retrieval_required=False),
        _golden("g3", retrieval_required=False),
    ]
    adapter = CallableAdapter(flaky)

    result = run(adapter, goldens, [metric])

    assert result.metadata.num_errors == 1
    by_id = {tr.golden_id: tr for tr in result.trajectory_results}
    assert by_id["g2"].error is not None
    assert "simulated API failure" in by_id["g2"].error
    assert by_id["g2"].trajectory_id is None
    assert by_id["g2"].metric_results == {}
    # the other two still scored normally
    assert by_id["g1"].error is None
    assert by_id["g3"].error is None
    # the failed record is excluded from the aggregate, not counted as a failure in it
    assert result.aggregate_scores["retrieval_necessity"]["total"] == 2


def test_run_all_records_fail_degenerate_case() -> None:
    def always_fails(question: str) -> dict:
        raise RuntimeError("down")

    goldens = [_golden("g1", retrieval_required=False)]
    adapter = CallableAdapter(always_fails)

    result = run(adapter, goldens, [metric])

    assert result.metadata.num_errors == 1
    assert result.aggregate_scores["retrieval_necessity"]["total"] == 0
    assert result.aggregate_scores["retrieval_necessity"]["necessity_score"] is None


def test_run_degenerate_empty_dataset() -> None:
    result = run(CallableAdapter(_always_answers_directly), [], [metric])
    assert result.metadata.num_trajectories == 0
    assert result.trajectory_results == []
    assert result.aggregate_scores["retrieval_necessity"]["total"] == 0


# ---------------------------------------------------------------------------
# run() — on_progress
# ---------------------------------------------------------------------------


def test_run_on_progress_called_once_per_golden_record_with_final_total() -> None:
    goldens = [_golden(f"g{i}", retrieval_required=False) for i in range(5)]
    calls: list[tuple[int, int]] = []

    run(
        CallableAdapter(_always_answers_directly),
        goldens,
        [metric],
        on_progress=lambda completed, total: calls.append((completed, total)),
    )

    assert len(calls) == 5
    # completion order isn't guaranteed under the thread pool, but every
    # call must report the same total, and completed counts 1..5 exactly
    # once each regardless of order.
    assert all(total == 5 for _completed, total in calls)
    assert sorted(completed for completed, _total in calls) == [1, 2, 3, 4, 5]


def test_run_on_progress_counts_adapter_failures_too() -> None:
    """A failed adapter call still finishes its slot in the run — progress
    must reach the full total even when every record errors."""

    def always_fails(question: str) -> dict:
        raise RuntimeError("down")

    goldens = [_golden(f"g{i}", retrieval_required=False) for i in range(3)]
    calls: list[tuple[int, int]] = []

    run(
        CallableAdapter(always_fails),
        goldens,
        [metric],
        on_progress=lambda completed, total: calls.append((completed, total)),
    )

    assert len(calls) == 3
    assert max(completed for completed, _total in calls) == 3


def test_run_degenerate_no_on_progress_does_not_raise() -> None:
    goldens = [_golden("g1", retrieval_required=False)]
    result = run(CallableAdapter(_always_answers_directly), goldens, [metric])
    assert result.metadata.num_trajectories == 1


def test_run_rejects_duplicate_golden_ids() -> None:
    goldens = [_golden("g1", retrieval_required=False), _golden("g1", retrieval_required=True)]
    with pytest.raises(ValueError, match="duplicate"):
        run(CallableAdapter(_always_answers_directly), goldens, [metric])


# ---------------------------------------------------------------------------
# save/load round trip
# ---------------------------------------------------------------------------


def test_save_and_load_run_result_round_trips(tmp_path: Path) -> None:
    goldens = [_golden("g1", retrieval_required=False)]
    result = run(CallableAdapter(_always_answers_directly), goldens, [metric])

    out_path = tmp_path / "run.json"
    save_run_result(result, out_path)
    restored = load_run_result(out_path)

    assert isinstance(restored, RunResult)
    assert restored == result


# ---------------------------------------------------------------------------
# load_golden_dataset
# ---------------------------------------------------------------------------


def test_load_golden_dataset_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    records = [_golden("g1", retrieval_required=True), _golden("g2", retrieval_required=False)]
    path.write_text(
        "\n".join(r.model_dump_json() for r in records) + "\n",
        encoding="utf-8",
    )

    loaded = load_golden_dataset(path)
    assert loaded == records


def test_load_golden_dataset_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    record = _golden("g1", retrieval_required=True)
    path.write_text(f"\n{record.model_dump_json()}\n\n", encoding="utf-8")

    loaded = load_golden_dataset(path)
    assert loaded == [record]


def test_load_golden_dataset_degenerate_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text("", encoding="utf-8")
    assert load_golden_dataset(path) == []


def test_load_golden_dataset_invalid_line_raises_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text('{"not": "a golden record"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="dataset.jsonl:1"):
        load_golden_dataset(path)


# ---------------------------------------------------------------------------
# calibration wiring
# ---------------------------------------------------------------------------


def _bad_retrieval_adapter(question: str) -> dict:
    return {
        "final_answer": "I don't know.",
        "steps": [
            {
                "step_type": "retrieval",
                "query": question,
                "chunks": [{"doc_id": "irrelevant", "text": "not relevant"}],
            },
            {"step_type": "answer", "text": "I don't know."},
        ],
    }


def test_run_populates_uncalibrated_state_for_judged_metric_with_no_labels() -> None:
    goldens = [
        GoldenRecord(
            id="g1",
            question="q",
            reference_answer="a",
            retrieval_required=True,
            min_steps=2,
            required_doc_ids=["doc-1"],
        )
    ]
    judge = FakeJudgeClient(lambda p: '{"outcome": "correctly_abstained", "reason": "test"}')
    result = run(CallableAdapter(_bad_retrieval_adapter), goldens, [RecoveryMetric(judge)])

    assert "recovery" in result.calibration
    assert result.calibration["recovery"].is_calibrated is False
    assert result.calibration["recovery"].n_labels == 0


def test_recalibrate_updates_saved_run_result_against_labels(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    goldens = [
        GoldenRecord(
            id="g1",
            question="q",
            reference_answer="a",
            retrieval_required=True,
            min_steps=2,
            required_doc_ids=["doc-1"],
        )
    ]
    judge = FakeJudgeClient(lambda p: '{"outcome": "correctly_abstained", "reason": "test"}')
    adapter = CallableAdapter(_bad_retrieval_adapter)

    saved = run(adapter, goldens, [RecoveryMetric(judge)])
    assert saved.calibration["recovery"].is_calibrated is False  # no labels yet

    trajectory_id = saved.trajectory_results[0].trajectory_id
    assert trajectory_id is not None
    append_label(
        labels_path,
        HumanLabel(
            trajectory_id=trajectory_id,
            golden_id="g1",
            metric_name="recovery",
            human_verdict="correctly_abstained",
            labeler="tester",
        ),
    )

    recalibrated = recalibrate(saved, labels_path)
    assert recalibrated.calibration["recovery"].n_labels == 1
    assert recalibrated.calibration["recovery"].kappa == 1.0
    assert (
        recalibrated.calibration["recovery"].is_calibrated is False
    )  # 1 < MIN_LABELS_FOR_CALIBRATION
    # recalibrate returns a new object; the original is untouched
    assert saved.calibration["recovery"].n_labels == 0


def test_recalibrate_degenerate_no_judged_metrics_returns_unchanged() -> None:
    goldens = [_golden("g1", retrieval_required=False)]
    result = run(CallableAdapter(_always_answers_directly), goldens, [metric])
    recalibrated = recalibrate(result, "does-not-matter.jsonl")
    assert recalibrated is result


# ---------------------------------------------------------------------------
# per-metric scoring failure isolation
# ---------------------------------------------------------------------------


def test_run_isolates_a_single_metric_failure_to_that_trajectory() -> None:
    """A judge returning unparseable JSON must not take down the whole run
    — only the one metric, on the one trajectory that triggered it."""
    golden_bad = GoldenRecord(
        id="g-bad",
        question="q",
        reference_answer="a",
        retrieval_required=True,
        min_steps=2,
        required_doc_ids=["doc-1"],
    )
    golden_fine = GoldenRecord(
        id="g-fine", question="q2", reference_answer="a", retrieval_required=False, min_steps=1
    )
    broken_judge = FakeJudgeClient(lambda p: "not valid json at all")

    def dispatch(question: str) -> dict:
        if question == "q":
            return _bad_retrieval_adapter(question)  # triggers a judge call -> parse failure
        return {"final_answer": "42", "steps": [{"step_type": "answer", "text": "42"}]}

    result = run(
        CallableAdapter(dispatch),
        [golden_bad, golden_fine],
        [RetrievalNecessityMetric(), RecoveryMetric(broken_judge)],
    )

    by_id = {tr.golden_id: tr for tr in result.trajectory_results}

    # the broken trajectory: no top-level error, trajectory intact, the
    # OTHER metric (retrieval_necessity) still scored fine, only recovery failed
    bad_tr = by_id["g-bad"]
    assert bad_tr.error is None
    assert bad_tr.trajectory is not None
    assert "retrieval_necessity" in bad_tr.metric_results
    assert "recovery" not in bad_tr.metric_results
    assert "recovery" in bad_tr.metric_errors
    assert "JudgeParseError" in bad_tr.metric_errors["recovery"]

    # the unrelated trajectory is completely unaffected
    fine_tr = by_id["g-fine"]
    assert fine_tr.metric_errors == {}
    assert "retrieval_necessity" in fine_tr.metric_results

    # the failed metric is excluded from its own aggregate, same as an
    # adapter-level failure would be — but retrieval_necessity's aggregate
    # still covers both trajectories since it wasn't affected
    assert result.aggregate_scores["retrieval_necessity"]["total"] == 2
    assert result.aggregate_scores["recovery"]["applicable"] == 0


def test_run_degenerate_all_metrics_fail_trajectory_still_recorded() -> None:
    broken_judge = FakeJudgeClient(lambda p: "not json")
    goldens = [
        GoldenRecord(
            id="g1",
            question="q",
            reference_answer="a",
            retrieval_required=True,
            min_steps=2,
            required_doc_ids=["doc-1"],
        )
    ]
    result = run(CallableAdapter(_bad_retrieval_adapter), goldens, [RecoveryMetric(broken_judge)])

    tr = result.trajectory_results[0]
    assert tr.error is None
    assert tr.trajectory is not None  # the trajectory itself was produced fine
    assert tr.metric_results == {}
    assert "recovery" in tr.metric_errors
    assert result.metadata.num_errors == 0  # this isn't an adapter-level error
    assert result.aggregate_scores["recovery"]["applicable"] == 0
