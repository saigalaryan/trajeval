"""Fast local linting for a golden dataset JSONL file.

`trajeval.runner.load_golden_dataset` deliberately raises on the *first*
malformed line — correct for `run()` itself, which shouldn't proceed past a
broken dataset at all. `validate_dataset` here is for a different moment:
before you spend the time (and, with judged metrics, the money) on a real
run, catch every problem in the file at once — every malformed line, not
just the first, plus dataset-level problems a single-line check can't see,
like duplicate ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from trajeval.types import GoldenRecord


@dataclass
class ValidationIssue:
    # None for a dataset-level issue that isn't tied to one line (currently
    # just duplicate ids — reported against the line where the duplicate
    # *reappears*, since the first occurrence was fine on its own).
    line: int | None
    message: str


@dataclass
class ValidationReport:
    path: str
    num_records: int
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def validate_dataset(path: str | Path) -> ValidationReport:
    """Parse every line of `path`, collecting every issue instead of
    stopping at the first one."""
    issues: list[ValidationIssue] = []
    records: list[GoldenRecord] = []
    first_seen_at: dict[str, int] = {}

    with Path(path).open(encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = GoldenRecord.model_validate_json(line)
            except Exception as exc:
                issues.append(ValidationIssue(line=line_no, message=str(exc)))
                continue

            if record.id in first_seen_at:
                issues.append(
                    ValidationIssue(
                        line=line_no,
                        message=(
                            f"duplicate id {record.id!r} "
                            f"(first seen on line {first_seen_at[record.id]})"
                        ),
                    )
                )
            else:
                first_seen_at[record.id] = line_no
            records.append(record)

    return ValidationReport(path=str(path), num_records=len(records), issues=issues)
