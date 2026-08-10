# Metrics

## Base

::: trajeval.metrics.base.Metric
::: trajeval.metrics.base.MetricResult

## Deterministic metrics

These never call a judge — pure functions of the trajectory and golden
record.

::: trajeval.metrics.termination.TerminationMetric
::: trajeval.metrics.trajectory_efficiency.TrajectoryEfficiencyMetric
::: trajeval.metrics.trajectory_efficiency.detect_loops
::: trajeval.metrics.retrieval_necessity.RetrievalNecessityMetric
::: trajeval.metrics.retrieval_necessity.RetrievalOutcome
::: trajeval.metrics.retrieval_necessity.classify

## LLM-judged metrics

Every judged metric must stay calibratable — see [Calibration](calibration.md).

::: trajeval.metrics.query_quality.QueryQualityMetric
::: trajeval.metrics.recovery.RecoveryMetric
::: trajeval.metrics.recovery.RecoveryOutcome
::: trajeval.metrics.faithfulness.FaithfulnessMetric

## Shared retrieval-context helpers

::: trajeval.metrics.context.relevant_doc_ids
::: trajeval.metrics.context.step_chunk_ids
::: trajeval.metrics.context.is_retrieval_adequate
