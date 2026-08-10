"""trajeval.calibration: the honesty layer for judged metrics.

Build a `HumanLabel` set with `trajeval label` (see `cli.run_labeling_session`),
then `compute_calibration` turns judge verdicts + human labels into a
`CalibrationState` — Cohen's kappa, overall and sliced by tag — that gets
attached to every `RunResult` a judged metric appears in.
"""

from trajeval.calibration.kappa import MIN_LABELS_FOR_CALIBRATION, cohens_kappa, compute_calibration
from trajeval.calibration.labels import JUDGED_METRIC_NAMES, HumanLabel, append_label, load_labels
from trajeval.calibration.verdicts import judge_verdict

__all__ = [
    "JUDGED_METRIC_NAMES",
    "MIN_LABELS_FOR_CALIBRATION",
    "HumanLabel",
    "append_label",
    "cohens_kappa",
    "compute_calibration",
    "judge_verdict",
    "load_labels",
]
