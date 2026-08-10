// TypeScript mirror of trajeval's RunResult JSON schema
// (packages/trajeval/src/trajeval/results.py, types.py). Hand-kept in sync
// deliberately — trajeval has no schema-export step, and duplicating a
// couple dozen fields by hand is simpler than adding a codegen pipeline for
// a schema that changes rarely. Only the fields the viewer actually reads
// are declared; unknown JSON fields pass through untyped.

export type StepType = "thought" | "retrieval" | "tool" | "answer";

export interface RetrievedChunk {
  doc_id: string;
  text: string;
  score: number | null;
  rank: number | null;
}

export interface ThoughtStep {
  step_type: "thought";
  text: string;
  timestamp: string | null;
}

export interface RetrievalStep {
  step_type: "retrieval";
  query: string;
  chunks: RetrievedChunk[];
  timestamp: string | null;
}

export interface ToolStep {
  step_type: "tool";
  tool_name: string;
  args: Record<string, unknown>;
  result: unknown;
  timestamp: string | null;
}

export interface AnswerStep {
  step_type: "answer";
  text: string;
  timestamp: string | null;
}

export type Step = ThoughtStep | RetrievalStep | ToolStep | AnswerStep;

export interface TrajectoryMetadata {
  model: string | null;
  timestamp: string | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  latency_ms: number | null;
  [key: string]: unknown;
}

export interface Trajectory {
  id: string;
  golden_id: string | null;
  question: string;
  final_answer: string;
  steps: Step[];
  metadata: TrajectoryMetadata;
}

export interface MetricResult {
  metric_name: string;
  value: number | null;
  details: Record<string, unknown>;
}

export interface TrajectoryResult {
  golden_id: string;
  question: string;
  tags: string[];
  trajectory_id: string | null;
  trajectory: Trajectory | null;
  metric_results: Record<string, MetricResult>;
  error: string | null;
  // One entry per metric that raised while scoring this trajectory —
  // distinct from `error` (an adapter failure: no trajectory produced at
  // all). See trajeval.results.TrajectoryResult.metric_errors.
  metric_errors: Record<string, string>;
}

export interface RunMetadata {
  run_id: string;
  started_at: string;
  finished_at: string;
  git_sha: string | null;
  config_hash: string;
  adapter_name: string;
  dataset_path: string | null;
  num_trajectories: number;
  num_errors: number;
  metric_names: string[];
}

export interface CalibrationState {
  is_calibrated: boolean;
  kappa: number | null;
  n_labels: number;
  kappa_by_tag: Record<string, number>;
}

export interface RunResult {
  schema_version: number;
  metadata: RunMetadata;
  trajectory_results: TrajectoryResult[];
  aggregate_scores: Record<string, Record<string, unknown>>;
  calibration: Record<string, CalibrationState>;
}

export const JUDGED_METRIC_NAMES = new Set([
  "query_quality",
  "recovery",
  "faithfulness",
]);

/** Loose runtime check — not full schema validation, just enough to catch
 * "this obviously isn't a RunResult" before the UI tries to render it. */
export function isRunResult(value: unknown): value is RunResult {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.schema_version === "number" &&
    typeof v.metadata === "object" &&
    v.metadata !== null &&
    Array.isArray(v.trajectory_results) &&
    typeof v.aggregate_scores === "object"
  );
}
