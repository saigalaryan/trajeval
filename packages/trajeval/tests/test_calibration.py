"""Tests for trajeval.calibration (labels, verdicts, kappa, labeling session)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from trajeval.calibration.cli import format_trajectory_for_labeling, run_labeling_session
from trajeval.calibration.kappa import cohens_kappa, compute_calibration
from trajeval.calibration.labels import HumanLabel, append_label, load_labels
from trajeval.calibration.verdicts import judge_verdict
from trajeval.metrics.base import MetricResult
from trajeval.results import CalibrationState, RunMetadata, RunResult, TrajectoryResult
from trajeval.types import (
    AnswerStep,
    GoldenRecord,
    RetrievalStep,
    RetrievedChunk,
    ThoughtStep,
    ToolStep,
    Trajectory,
)

# ---------------------------------------------------------------------------
# labels: storage round trip
# ---------------------------------------------------------------------------


def _label(**overrides) -> HumanLabel:
    defaults = dict(
        trajectory_id="t1",
        golden_id="g1",
        metric_name="recovery",
        human_verdict="correctly_abstained",
        labeler="arush",
        tags=["easy"],
    )
    defaults.update(overrides)
    return HumanLabel(**defaults)


def test_load_labels_degenerate_missing_file(tmp_path: Path) -> None:
    assert load_labels(tmp_path / "nope.jsonl") == []


def test_append_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    append_label(path, _label())
    append_label(path, _label(trajectory_id="t2"))
    loaded = load_labels(path)
    assert len(loaded) == 2
    assert loaded[0].trajectory_id == "t1"
    assert loaded[1].trajectory_id == "t2"


def test_append_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "labels.jsonl"
    append_label(path, _label())
    assert path.exists()


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


def test_judge_verdict_recovery_judged_branch() -> None:
    result = MetricResult(
        metric_name="recovery", value=0.0, details={"outcome": "answered_from_bad_context"}
    )
    assert judge_verdict("recovery", result) == "answered_from_bad_context"


def test_judge_verdict_recovery_reformulated_branch_not_calibratable() -> None:
    """reformulated is deterministic, never called the judge — nothing to calibrate."""
    result = MetricResult(metric_name="recovery", value=1.0, details={"outcome": "reformulated"})
    assert judge_verdict("recovery", result) is None


def test_judge_verdict_recovery_not_applicable() -> None:
    result = MetricResult(metric_name="recovery", value=None, details={})
    assert judge_verdict("recovery", result) is None


def test_judge_verdict_query_quality_rounds_mean() -> None:
    result = MetricResult(
        metric_name="query_quality",
        value=1.0,
        details={
            "queries": [
                {"query": "a", "hit": True, "judged_quality": 5},
                {"query": "b", "hit": False, "judged_quality": 5},
            ]
        },
    )
    assert judge_verdict("query_quality", result) == "5"


def test_judge_verdict_faithfulness_fully_supported() -> None:
    result = MetricResult(
        metric_name="faithfulness",
        value=1.0,
        details={"claims": [{"claim": "x", "supported": True}, {"claim": "y", "supported": True}]},
    )
    assert judge_verdict("faithfulness", result) == "fully_supported"


def test_judge_verdict_faithfulness_not_fully_supported() -> None:
    result = MetricResult(
        metric_name="faithfulness",
        value=0.5,
        details={"claims": [{"claim": "x", "supported": True}, {"claim": "y", "supported": False}]},
    )
    assert judge_verdict("faithfulness", result) == "not_fully_supported"


def test_judge_verdict_unknown_metric_returns_none() -> None:
    result = MetricResult(
        metric_name="retrieval_necessity", value=1.0, details={"outcome": "correct_search"}
    )
    assert judge_verdict("retrieval_necessity", result) is None


# ---------------------------------------------------------------------------
# cohens_kappa
# ---------------------------------------------------------------------------


def test_kappa_perfect_agreement() -> None:
    judge = ["a", "b", "a", "b"]
    human = ["a", "b", "a", "b"]
    assert cohens_kappa(judge, human) == 1.0


def test_kappa_no_better_than_chance_is_near_zero() -> None:
    # constructed so observed agreement equals expected chance agreement
    judge = ["a", "a", "b", "b"]
    human = ["a", "b", "a", "b"]
    kappa = cohens_kappa(judge, human)
    assert -0.1 < kappa < 0.1


def test_kappa_systematic_disagreement_is_negative() -> None:
    judge = ["a", "a", "a", "a"]
    human = ["b", "b", "b", "b"]
    # every category count for judge is all "a", human all "b": pe = 0*... -> 0
    # po = 0 -> kappa = (0-0)/(1-0) = 0, not negative; use a case with real disagreement instead
    kappa = cohens_kappa(judge, human)
    assert kappa == 0.0


def test_kappa_degenerate_empty_raises() -> None:
    with pytest.raises(ValueError):
        cohens_kappa([], [])


def test_kappa_degenerate_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError):
        cohens_kappa(["a"], ["a", "b"])


def test_kappa_single_category_perfect_agreement_no_div_by_zero() -> None:
    assert cohens_kappa(["a", "a", "a"], ["a", "a", "a"]) == 1.0


# ---------------------------------------------------------------------------
# weighted kappa (ordinal categories, e.g. query_quality's "1".."5")
# ---------------------------------------------------------------------------


def test_kappa_weighted_penalizes_near_misses_less_than_far_misses() -> None:
    """A judge/human gap of one category (4 vs 5) should hurt less than a
    gap of four (1 vs 5) — the entire point of weighting ordinal labels.

    Both cases include "anchor" pairs spanning the full 1-5 range with
    perfect agreement, so the category set — and therefore the (k-1)
    normalizer — is identical between the two comparisons. Without the
    anchors, only 2 distinct categories would ever appear in either call,
    and quadratic distance always normalizes to 1 with just 2 categories —
    the raw numeric gap (1 vs 4) would never actually show up.
    """
    # Anchors alone span all five categories with perfect agreement, so the
    # category set — {1,2,3,4,5}, k=5 — is identical whichever divergent
    # pair gets appended below; only the divergent pair's distance differs.
    anchors_judge = ["1", "2", "3", "4", "5"]
    anchors_human = ["1", "2", "3", "4", "5"]

    near_miss = cohens_kappa(
        anchors_judge + ["5", "5", "5"], anchors_human + ["4", "4", "4"], weights="quadratic"
    )
    far_miss = cohens_kappa(
        anchors_judge + ["5", "5", "5"], anchors_human + ["1", "1", "1"], weights="quadratic"
    )
    assert near_miss > far_miss


def test_kappa_weighted_perfect_agreement_still_one() -> None:
    assert cohens_kappa(["1", "3", "5"], ["1", "3", "5"], weights="quadratic") == 1.0
    assert cohens_kappa(["1", "3", "5"], ["1", "3", "5"], weights="linear") == 1.0


def test_kappa_none_and_quadratic_agree_when_only_two_categories() -> None:
    """With exactly two categories, "distance" can only ever be 0 or 1 —
    weighting can't distinguish anything, so quadratic must reduce to
    unweighted kappa exactly."""
    judge = ["1", "5", "1", "5", "1"]
    human = ["1", "1", "5", "5", "1"]
    assert cohens_kappa(judge, human, weights="quadratic") == cohens_kappa(judge, human)


def test_kappa_category_sort_key_is_numeric_not_lexicographic() -> None:
    """query_quality's categories are single digits ("1".."5"), where
    numeric and lexicographic ordering happen to coincide — so this asserts
    the sort key directly rather than via a kappa value, to actually catch
    a regression to plain string sorting (which would put "10" before "2")."""
    from trajeval.calibration.kappa import _category_sort_key

    assert sorted(["10", "2", "1"], key=_category_sort_key) == ["1", "2", "10"]


def test_kappa_weights_fall_back_to_lexicographic_for_non_numeric_categories() -> None:
    """recovery/faithfulness categories aren't numeric at all — the sort key
    must degrade gracefully (not raise) rather than assume every metric's
    verdicts are ordinal."""
    from trajeval.calibration.kappa import _category_sort_key

    ordered = sorted(["correctly_abstained", "answered_from_bad_context"], key=_category_sort_key)
    assert ordered == ["answered_from_bad_context", "correctly_abstained"]


def test_compute_calibration_uses_quadratic_weights_for_query_quality() -> None:
    """Proof `compute_calibration` actually wires the metric-name ->
    weighting map through for query_quality: its output must match calling
    `cohens_kappa` directly with `weights="quadratic"` on the identical
    paired sequence — and must therefore differ from the unweighted value,
    or this test would pass even if the wiring were silently missing.
    """
    judged_qualities = [1, 2, 3, 4, 5]
    human_verdicts = ["1", "2", "3", "5", "5"]  # t3 is a near-miss (4 vs 5)

    trajectory_results = [
        TrajectoryResult(
            golden_id=f"g{i}",
            question="q",
            trajectory_id=f"t{i}",
            metric_results={
                "query_quality": MetricResult(
                    metric_name="query_quality",
                    value=1.0,
                    details={"queries": [{"query": "x", "hit": True, "judged_quality": judged}]},
                )
            },
        )
        for i, judged in enumerate(judged_qualities)
    ]
    metadata = RunMetadata(
        run_id="r",
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        config_hash="x",
        adapter_name="a",
        num_trajectories=len(judged_qualities),
        num_errors=0,
        metric_names=["query_quality"],
    )
    run_result = RunResult(
        metadata=metadata, trajectory_results=trajectory_results, aggregate_scores={}
    )
    labels = [
        _label(trajectory_id=f"t{i}", metric_name="query_quality", human_verdict=h)
        for i, h in enumerate(human_verdicts)
    ]

    state = compute_calibration(run_result, labels, "query_quality", min_labels=len(labels))

    judge_labels = [str(j) for j in judged_qualities]
    expected_weighted = cohens_kappa(judge_labels, human_verdicts, weights="quadratic")
    expected_unweighted = cohens_kappa(judge_labels, human_verdicts)

    assert state.kappa == expected_weighted
    assert state.kappa != expected_unweighted


# ---------------------------------------------------------------------------
# compute_calibration
# ---------------------------------------------------------------------------


def _run_result_with_recovery(outcomes: dict[str, str]) -> RunResult:
    trajectory_results = [
        TrajectoryResult(
            golden_id=f"g-{tid}",
            question="q",
            trajectory_id=tid,
            metric_results={
                "recovery": MetricResult(
                    metric_name="recovery", value=1.0, details={"outcome": outcome}
                )
            },
        )
        for tid, outcome in outcomes.items()
    ]
    metadata = RunMetadata(
        run_id="r1",
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        config_hash="x",
        adapter_name="a",
        num_trajectories=len(outcomes),
        num_errors=0,
        metric_names=["recovery"],
    )
    return RunResult(metadata=metadata, trajectory_results=trajectory_results, aggregate_scores={})


def test_compute_calibration_below_threshold_is_uncalibrated() -> None:
    run_result = _run_result_with_recovery(
        {"t1": "correctly_abstained", "t2": "answered_from_bad_context"}
    )
    labels = [
        _label(trajectory_id="t1", human_verdict="correctly_abstained"),
        _label(trajectory_id="t2", human_verdict="answered_from_bad_context"),
    ]
    state = compute_calibration(run_result, labels, "recovery", min_labels=50)
    assert state.n_labels == 2
    assert state.is_calibrated is False
    assert state.kappa == 1.0  # still computed and reported, just not "trusted"


def test_compute_calibration_meets_threshold_when_enough_labels() -> None:
    outcomes = {f"t{i}": "correctly_abstained" for i in range(3)}
    run_result = _run_result_with_recovery(outcomes)
    labels = [_label(trajectory_id=tid, human_verdict="correctly_abstained") for tid in outcomes]
    state = compute_calibration(run_result, labels, "recovery", min_labels=3)
    assert state.is_calibrated is True
    assert state.n_labels == 3
    assert state.kappa == 1.0


def test_compute_calibration_degenerate_no_labels() -> None:
    run_result = _run_result_with_recovery({"t1": "correctly_abstained"})
    state = compute_calibration(run_result, [], "recovery")
    assert isinstance(state, CalibrationState)
    assert state.is_calibrated is False
    assert state.n_labels == 0
    assert state.kappa is None


def test_compute_calibration_slices_by_tag() -> None:
    outcomes = {
        "t1": "correctly_abstained",
        "t2": "correctly_abstained",
        "t3": "answered_from_bad_context",
    }
    run_result = _run_result_with_recovery(outcomes)
    labels = [
        _label(trajectory_id="t1", human_verdict="correctly_abstained", tags=["easy"]),
        _label(trajectory_id="t2", human_verdict="correctly_abstained", tags=["easy"]),
        _label(trajectory_id="t3", human_verdict="answered_from_bad_context", tags=["hard"]),
    ]
    state = compute_calibration(run_result, labels, "recovery", min_labels=3)
    assert "easy" in state.kappa_by_tag
    assert state.kappa_by_tag["easy"] == 1.0
    # "hard" has only 1 label -> too few to slice, excluded
    assert "hard" not in state.kappa_by_tag


def test_compute_calibration_excludes_labels_for_untracked_trajectories() -> None:
    run_result = _run_result_with_recovery({"t1": "correctly_abstained"})
    labels = [_label(trajectory_id="does-not-exist", human_verdict="correctly_abstained")]
    state = compute_calibration(run_result, labels, "recovery")
    assert state.n_labels == 0


# ---------------------------------------------------------------------------
# labeling session
# ---------------------------------------------------------------------------


def test_format_trajectory_for_labeling_never_reveals_judge_verdict() -> None:
    golden = GoldenRecord(
        id="g1",
        question="What year?",
        reference_answer="1889",
        retrieval_required=True,
        min_steps=2,
    )
    trajectory = Trajectory(
        question="What year?",
        final_answer="1889",
        steps=[RetrievalStep(query="q", chunks=[]), AnswerStep(text="1889")],
    )
    text = format_trajectory_for_labeling(trajectory, golden)
    assert "What year?" in text
    assert "1889" in text
    assert "outcome" not in text.lower()
    assert "verdict" not in text.lower()


def test_format_trajectory_for_labeling_renders_every_step_type() -> None:
    golden = GoldenRecord(
        id="g1",
        question="What year?",
        reference_answer="1889",
        retrieval_required=True,
        min_steps=2,
    )
    long_chunk_text = "a" * 250
    trajectory = Trajectory(
        question="What year?",
        final_answer="1889",
        steps=[
            ThoughtStep(text="let me think about this"),
            RetrievalStep(
                query="eiffel tower built",
                chunks=[RetrievedChunk(doc_id="d1", text=long_chunk_text, score=0.9, rank=1)],
            ),
            ToolStep(tool_name="calculator", args={"x": 1}, result=2),
            AnswerStep(text="1889"),
        ],
    )

    text = format_trajectory_for_labeling(trajectory, golden)

    assert "thought: let me think about this" in text
    assert "retrieval: query='eiffel tower built'" in text
    assert "[d1]" in text
    assert text.count("...") == 1  # the 250-char chunk got truncated at 200
    assert "tool: calculator({'x': 1}) -> 2" in text
    assert "Final answer: 1889" in text


def test_run_labeling_session_records_scripted_answers(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    run_result = _run_result_with_recovery(
        {"t1": "correctly_abstained", "t2": "answered_from_bad_context"}
    )
    goldens = {
        "g-t1": GoldenRecord(
            id="g-t1", question="q1", reference_answer="a", retrieval_required=True, min_steps=1
        ),
        "g-t2": GoldenRecord(
            id="g-t2", question="q2", reference_answer="a", retrieval_required=True, min_steps=1
        ),
    }
    trajectories = {
        "t1": Trajectory(question="q1", final_answer="a", steps=[AnswerStep(text="a")]),
        "t2": Trajectory(question="q2", final_answer="a", steps=[AnswerStep(text="a")]),
    }

    scripted = iter(["correctly_abstained", "answered_from_bad_context"])
    printed: list[str] = []

    recorded = run_labeling_session(
        run_result,
        trajectories,
        goldens,
        "recovery",
        n=10,
        labeler="tester",
        labels_path=labels_path,
        input_fn=lambda prompt: next(scripted),
        print_fn=printed.append,
    )

    assert recorded == 2
    saved = load_labels(labels_path)
    assert len(saved) == 2
    assert {label.trajectory_id for label in saved} == {"t1", "t2"}


def test_run_labeling_session_respects_n_limit(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    run_result = _run_result_with_recovery(
        {
            "t1": "correctly_abstained",
            "t2": "answered_from_bad_context",
            "t3": "correctly_abstained",
        }
    )
    goldens = {
        f"g-{tid}": GoldenRecord(
            id=f"g-{tid}", question="q", reference_answer="a", retrieval_required=True, min_steps=1
        )
        for tid in ("t1", "t2", "t3")
    }
    trajectories = {
        tid: Trajectory(question="q", final_answer="a", steps=[AnswerStep(text="a")])
        for tid in ("t1", "t2", "t3")
    }
    recorded = run_labeling_session(
        run_result,
        trajectories,
        goldens,
        "recovery",
        n=2,
        labeler="tester",
        labels_path=labels_path,
        input_fn=lambda prompt: "correctly_abstained",
        print_fn=lambda s: None,
    )
    assert recorded == 2


def test_run_labeling_session_skips_already_labeled_by_same_labeler(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    append_label(labels_path, _label(trajectory_id="t1", labeler="tester"))
    run_result = _run_result_with_recovery({"t1": "correctly_abstained"})
    goldens = {
        "g-t1": GoldenRecord(
            id="g-t1", question="q", reference_answer="a", retrieval_required=True, min_steps=1
        )
    }
    trajectories = {"t1": Trajectory(question="q", final_answer="a", steps=[AnswerStep(text="a")])}

    recorded = run_labeling_session(
        run_result,
        trajectories,
        goldens,
        "recovery",
        n=10,
        labeler="tester",
        labels_path=labels_path,
        input_fn=lambda prompt: "correctly_abstained",
        print_fn=lambda s: None,
    )
    assert recorded == 0


def test_run_labeling_session_reprompts_on_invalid_input(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    run_result = _run_result_with_recovery({"t1": "correctly_abstained"})
    goldens = {
        "g-t1": GoldenRecord(
            id="g-t1", question="q", reference_answer="a", retrieval_required=True, min_steps=1
        )
    }
    trajectories = {"t1": Trajectory(question="q", final_answer="a", steps=[AnswerStep(text="a")])}

    scripted = iter(["not-a-valid-option", "correctly_abstained"])
    recorded = run_labeling_session(
        run_result,
        trajectories,
        goldens,
        "recovery",
        n=10,
        labeler="tester",
        labels_path=labels_path,
        input_fn=lambda prompt: next(scripted),
        print_fn=lambda s: None,
    )
    assert recorded == 1


def test_run_labeling_session_degenerate_no_candidates(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    run_result = _run_result_with_recovery({})
    recorded = run_labeling_session(
        run_result,
        {},
        {},
        "recovery",
        n=10,
        labeler="tester",
        labels_path=labels_path,
        input_fn=lambda p: "correctly_abstained",
        print_fn=lambda s: None,
    )
    assert recorded == 0


def test_run_labeling_session_unknown_metric_raises(tmp_path: Path) -> None:
    run_result = _run_result_with_recovery({})
    with pytest.raises(ValueError, match="not a judged metric"):
        run_labeling_session(
            run_result,
            {},
            {},
            "retrieval_necessity",
            n=1,
            labeler="t",
            labels_path=tmp_path / "labels.jsonl",
        )


def test_run_labeling_session_skips_candidates_with_no_trajectory_id(tmp_path: Path) -> None:
    """A TrajectoryResult can exist with trajectory_id=None (the adapter
    call failed — see runner.run()'s error handling); it must be silently
    skipped as a labeling candidate, not crash the session."""
    labels_path = tmp_path / "labels.jsonl"
    run_result = _run_result_with_recovery({"t1": "correctly_abstained"})
    run_result.trajectory_results.append(
        TrajectoryResult(
            golden_id="g-no-id",
            question="q",
            trajectory_id=None,
            metric_results={
                "recovery": MetricResult(
                    metric_name="recovery", value=1.0, details={"outcome": "correctly_abstained"}
                )
            },
        )
    )
    goldens = {
        "g-t1": GoldenRecord(
            id="g-t1", question="q", reference_answer="a", retrieval_required=True, min_steps=1
        )
    }
    trajectories = {"t1": Trajectory(question="q", final_answer="a", steps=[AnswerStep(text="a")])}

    recorded = run_labeling_session(
        run_result,
        trajectories,
        goldens,
        "recovery",
        n=10,
        labeler="tester",
        labels_path=labels_path,
        input_fn=lambda p: "correctly_abstained",
        print_fn=lambda s: None,
    )
    assert recorded == 1


def test_run_labeling_session_skips_trajectories_without_a_judge_verdict(tmp_path: Path) -> None:
    """No metric_result at all for this metric, and a metric_result whose
    outcome never went through the judge (deterministic "reformulated"
    branch — see judge_verdict) both mean "nothing to label here"."""
    labels_path = tmp_path / "labels.jsonl"
    run_result = _run_result_with_recovery({"t1": "correctly_abstained"})
    run_result.trajectory_results.append(
        TrajectoryResult(golden_id="g-t2", question="q", trajectory_id="t2", metric_results={})
    )
    run_result.trajectory_results.append(
        TrajectoryResult(
            golden_id="g-t3",
            question="q",
            trajectory_id="t3",
            metric_results={
                "recovery": MetricResult(
                    metric_name="recovery", value=1.0, details={"outcome": "reformulated"}
                )
            },
        )
    )
    goldens = {
        f"g-{tid}": GoldenRecord(
            id=f"g-{tid}", question="q", reference_answer="a", retrieval_required=True, min_steps=1
        )
        for tid in ("t1", "t2", "t3")
    }
    trajectories = {
        tid: Trajectory(question="q", final_answer="a", steps=[AnswerStep(text="a")])
        for tid in ("t1", "t2", "t3")
    }

    recorded = run_labeling_session(
        run_result,
        trajectories,
        goldens,
        "recovery",
        n=10,
        labeler="tester",
        labels_path=labels_path,
        input_fn=lambda p: "correctly_abstained",
        print_fn=lambda s: None,
    )
    assert recorded == 1


def test_run_labeling_session_skips_candidate_missing_trajectory_or_golden(
    tmp_path: Path,
) -> None:
    """A candidate can pass the metric/verdict filter yet still have no
    matching entry in the `trajectories`/`goldens` dicts the caller passed
    in (a caller bug, or stale data) — skipped, not a KeyError."""
    labels_path = tmp_path / "labels.jsonl"
    run_result = _run_result_with_recovery(
        {"t1": "correctly_abstained", "t2": "correctly_abstained"}
    )
    goldens = {
        "g-t1": GoldenRecord(
            id="g-t1", question="q", reference_answer="a", retrieval_required=True, min_steps=1
        )
        # g-t2 deliberately missing
    }
    trajectories = {
        "t1": Trajectory(question="q", final_answer="a", steps=[AnswerStep(text="a")])
        # t2 deliberately missing
    }

    recorded = run_labeling_session(
        run_result,
        trajectories,
        goldens,
        "recovery",
        n=10,
        labeler="tester",
        labels_path=labels_path,
        input_fn=lambda p: "correctly_abstained",
        print_fn=lambda s: None,
    )
    assert recorded == 1
