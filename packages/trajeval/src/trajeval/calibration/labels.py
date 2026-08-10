"""Human labels for judge calibration.

One `HumanLabel` is one person's independent judgment of the same categorical
decision a judged metric made on one trajectory — see
`trajeval.calibration.verdicts` for how a metric's `MetricResult.details` is
reduced to the single category a human can be asked to reproduce blind
(without seeing what the judge said, to keep the comparison honest).

Storage is append-only JSONL, versioned like every other trajeval artifact,
so a label file is diffable and safe to concatenate across labeling
sessions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from trajeval.types import SCHEMA_VERSION, TrajevalModel

JUDGED_METRIC_NAMES = frozenset({"query_quality", "recovery", "faithfulness"})


class HumanLabel(TrajevalModel):
    schema_version: int = SCHEMA_VERSION
    trajectory_id: str
    golden_id: str
    metric_name: str
    human_verdict: str
    labeler: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Copied from the golden record at label time, so kappa can be sliced by
    # tag without re-joining against the original dataset later.
    tags: list[str] = Field(default_factory=list)


def load_labels(path: str | Path) -> list[HumanLabel]:
    """Load a JSONL label file. Missing file returns an empty list — no
    labels yet is a valid, common starting state, not an error."""
    p = Path(path)
    if not p.exists():
        return []
    labels: list[HumanLabel] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(HumanLabel.model_validate_json(line))
    return labels


def append_label(path: str | Path, label: HumanLabel) -> None:
    """Append one label to the JSONL file, creating it if necessary."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(label.model_dump_json() + "\n")
