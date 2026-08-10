"""trajeval metrics: score trajectories against golden records.

`retrieval_necessity`, `trajectory_efficiency`, and `termination` are fully
deterministic. `query_quality`, `recovery`, and `faithfulness` are judged and
require a `trajeval.judge.JudgeClient` — see `trajeval.calibration` before
trusting their scores; an uncalibrated judged metric should be treated as
informative, not authoritative.
"""

from trajeval.metrics.base import Metric, MetricResult
from trajeval.metrics.context import is_retrieval_adequate, relevant_doc_ids, step_chunk_ids
from trajeval.metrics.faithfulness import FaithfulnessMetric
from trajeval.metrics.query_quality import QueryQualityMetric
from trajeval.metrics.recovery import RecoveryMetric, RecoveryOutcome
from trajeval.metrics.retrieval_necessity import (
    RetrievalNecessityMetric,
    RetrievalOutcome,
    classify,
)
from trajeval.metrics.termination import TerminationMetric
from trajeval.metrics.trajectory_efficiency import TrajectoryEfficiencyMetric, detect_loops

__all__ = [
    "FaithfulnessMetric",
    "Metric",
    "MetricResult",
    "QueryQualityMetric",
    "RecoveryMetric",
    "RecoveryOutcome",
    "RetrievalNecessityMetric",
    "RetrievalOutcome",
    "TerminationMetric",
    "TrajectoryEfficiencyMetric",
    "classify",
    "detect_loops",
    "is_retrieval_adequate",
    "relevant_doc_ids",
    "step_chunk_ids",
]
