"use client";

import { useStore } from "@/lib/store";
import { JUDGED_METRIC_NAMES } from "@/lib/types";
import { fmtValue } from "@/lib/format";
import { CalibrationBadge } from "@/components/CalibrationBadge";
import { EmptyState } from "@/components/EmptyState";

export default function RunOverviewPage() {
  const { primary, primaryName } = useStore();
  if (!primary) return <EmptyState />;

  const { metadata, aggregate_scores, calibration } = primary;

  return (
    <div>
      <h1 className="mb-1 text-lg font-semibold">{primaryName}</h1>
      <dl className="mb-6 grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-xs text-neutral-500 sm:grid-cols-4">
        <Field label="run id" value={metadata.run_id} />
        <Field label="adapter" value={metadata.adapter_name} />
        <Field label="trajectories" value={String(metadata.num_trajectories)} />
        <Field label="errors" value={String(metadata.num_errors)} />
        <Field label="git sha" value={metadata.git_sha ?? "n/a"} />
        <Field label="dataset" value={metadata.dataset_path ?? "n/a"} />
        <Field label="started" value={metadata.started_at} />
        <Field label="finished" value={metadata.finished_at} />
      </dl>

      <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-neutral-500">
        Aggregate scores
      </h2>
      <div className="space-y-4">
        {Object.entries(aggregate_scores).map(([metricName, agg]) => (
          <div
            key={metricName}
            className="rounded border border-neutral-200 p-3 dark:border-neutral-800"
          >
            <div className="mb-2 flex items-center gap-2">
              <span className="font-mono text-sm font-medium">{metricName}</span>
              {JUDGED_METRIC_NAMES.has(metricName) && (
                <CalibrationBadge state={calibration[metricName]} />
              )}
            </div>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {Object.entries(agg).map(([key, value]) => (
                <div key={key} className="text-xs">
                  <div className="text-neutral-500">{key}</div>
                  <div className="font-mono font-medium">{fmtValue(value)}</div>
                </div>
              ))}
            </div>
          </div>
        ))}
        {Object.keys(aggregate_scores).length === 0 && (
          <p className="text-sm text-neutral-500">No metrics recorded on this run.</p>
        )}
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-neutral-400">{label}: </span>
      <span className="text-neutral-700 dark:text-neutral-300">{value}</span>
    </div>
  );
}
