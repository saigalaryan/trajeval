"""Tests for trajeval.dataset_validation."""

from __future__ import annotations

from pathlib import Path

from trajeval.dataset_validation import validate_dataset

_VALID_LINE = (
    '{{"id": "{id_}", "question": "q", "reference_answer": "a", '
    '"retrieval_required": false, "min_steps": 1}}'
)


def test_validate_dataset_clean_file_has_no_issues(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        _VALID_LINE.format(id_="g1") + "\n" + _VALID_LINE.format(id_="g2") + "\n",
        encoding="utf-8",
    )

    report = validate_dataset(path)

    assert report.ok
    assert report.num_records == 2
    assert report.issues == []


def test_validate_dataset_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n" + _VALID_LINE.format(id_="g1") + "\n\n", encoding="utf-8")

    report = validate_dataset(path)

    assert report.ok
    assert report.num_records == 1


def test_validate_dataset_reports_malformed_json_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        _VALID_LINE.format(id_="g1") + "\nnot valid json\n" + _VALID_LINE.format(id_="g2") + "\n",
        encoding="utf-8",
    )

    report = validate_dataset(path)

    assert not report.ok
    assert report.num_records == 2  # the two good lines still parsed
    assert len(report.issues) == 1
    assert report.issues[0].line == 2


def test_validate_dataset_reports_missing_required_field(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    # missing "min_steps"
    path.write_text(
        '{"id": "g1", "question": "q", "reference_answer": "a", "retrieval_required": false}\n',
        encoding="utf-8",
    )

    report = validate_dataset(path)

    assert not report.ok
    assert report.issues[0].line == 1
    assert "min_steps" in report.issues[0].message


def test_validate_dataset_reports_duplicate_id_against_the_reappearing_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        _VALID_LINE.format(id_="g1")
        + "\n"
        + _VALID_LINE.format(id_="g2")
        + "\n"
        + _VALID_LINE.format(id_="g1")
        + "\n",
        encoding="utf-8",
    )

    report = validate_dataset(path)

    assert not report.ok
    assert len(report.issues) == 1
    issue = report.issues[0]
    assert issue.line == 3
    assert "g1" in issue.message
    assert "line 1" in issue.message


def test_validate_dataset_collects_every_issue_not_just_the_first(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        "bad line 1\nbad line 2\n" + _VALID_LINE.format(id_="g1") + "\n",
        encoding="utf-8",
    )

    report = validate_dataset(path)

    assert len(report.issues) == 2
    assert [i.line for i in report.issues] == [1, 2]


def test_validate_dataset_degenerate_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text("", encoding="utf-8")

    report = validate_dataset(path)

    assert report.ok
    assert report.num_records == 0
