"use client";

import { useMemo } from "react";
import { useStore } from "@/lib/store";
import { fmtValue } from "@/lib/format";
import { deltaSentiment } from "@/lib/metricDirection";
import { FileLoader } from "@/components/FileLoader";
import { EmptyState } from "@/components/EmptyState";
import type { RunResult } from "@/lib/types";

const SENTIMENT_CLASS: Record<ReturnType<typeof deltaSentiment>, string> = {
  better: "text-emerald-600 dark:text-emerald-400",
  worse: "text-red-600 dark:text-red-400",
  neutral: "",
};

interface Delta {
  metricName: string;
  key: string;
  baseline: unknown;
  candidate: unknown;
  delta: number | null;
}

function diffAggregates(baseline: RunResult, candidate: RunResult): Delta[] {
  const metricNames = new Set([
    ...Object.keys(baseline.aggregate_scores),
    ...Object.keys(candidate.aggregate_scores),
  ]);
  const deltas: Delta[] = [];
  for (const metricName of metricNames) {
    const baseAgg = baseline.aggregate_scores[metricName] ?? {};
    const candAgg = candidate.aggregate_scores[metricName] ?? {};
    const keys = new Set([...Object.keys(baseAgg), ...Object.keys(candAgg)]);
    for (const key of keys) {
      const b = baseAgg[key];
      const c = candAgg[key];
      const bNum = typeof b === "number" ? b : null;
      const cNum = typeof c === "number" ? c : null;
      deltas.push({
        metricName,
        key,
        baseline: b,
        candidate: c,
        delta: bNum !== null && cNum !== null ? cNum - bNum : null,
      });
    }
  }
  return deltas;
}

/** Trajectories present in both runs (matched by golden_id) whose per-metric
 * pass/fail verdict (value >= 1 vs < 1) flipped between baseline and
 * candidate — the "what actually changed" list, not just the aggregate. */
function changedVerdicts(baseline: RunResult, candidate: RunResult) {
  const baseByGolden = new Map(baseline.trajectory_results.map((tr) => [tr.golden_id, tr]));
  const changes: { golden_id: string; question: string; metric: string; from: unknown; to: unknown }[] = [];
  for (const candTr of candidate.trajectory_results) {
    const baseTr = baseByGolden.get(candTr.golden_id);
    if (!baseTr) continue;
    for (const [metric, candResult] of Object.entries(candTr.metric_results)) {
      const baseResult = baseTr.metric_results[metric];
      if (!baseResult) continue;
      const basePass = baseResult.value !== null && baseResult.value >= 1;
      const candPass = candResult.value !== null && candResult.value >= 1;
      if (basePass !== candPass) {
        changes.push({
          golden_id: candTr.golden_id,
          question: candTr.question,
          metric,
          from: baseResult.value,
          to: candResult.value,
        });
      }
    }
  }
  return changes;
}

export default function ComparePage() {
  const { primary, primaryName, secondary, secondaryName, loadSecondary } = useStore();

  const deltas = useMemo(
    () => (primary && secondary ? diffAggregates(primary, secondary) : []),
    [primary, secondary]
  );
  const changes = useMemo(
    () => (primary && secondary ? changedVerdicts(primary, secondary) : []),
    [primary, secondary]
  );

  if (!primary) return <EmptyState />;

  if (!secondary) {
    return (
      <div className="mx-auto max-w-xl">
        <h1 className="mb-1 text-lg font-semibold">Compare</h1>
        <p className="mb-4 text-sm text-neutral-500">
          Baseline: <span className="font-mono">{primaryName}</span>. Load a second run to compare
          against it.
        </p>
        <FileLoader label="Load candidate RunResult JSON" onLoaded={loadSecondary} />
      </div>
    );
  }

  return (
    <div>
      <h1 className="mb-1 text-lg font-semibold">Compare</h1>
      <p className="mb-6 text-sm text-neutral-500">
        <span className="font-mono">{primaryName}</span> (baseline) vs{" "}
        <span className="font-mono">{secondaryName}</span> (candidate)
      </p>

      <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-neutral-500">
        Aggregate deltas
      </h2>
      <table className="mb-8 w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-neutral-200 text-left text-xs text-neutral-500 dark:border-neutral-800">
            <th className="py-1 pr-3 font-normal">Metric</th>
            <th className="py-1 pr-3 font-normal">Key</th>
            <th className="py-1 pr-3 font-normal">Baseline</th>
            <th className="py-1 pr-3 font-normal">Candidate</th>
            <th className="py-1 pr-3 font-normal">Delta</th>
          </tr>
        </thead>
        <tbody>
          {deltas.map((d, i) => (
            <tr key={i} className="border-b border-neutral-100 font-mono dark:border-neutral-900">
              <td className="py-1 pr-3">{d.metricName}</td>
              <td className="py-1 pr-3 text-neutral-500">{d.key}</td>
              <td className="py-1 pr-3">{fmtValue(d.baseline)}</td>
              <td className="py-1 pr-3">{fmtValue(d.candidate)}</td>
              <td className={"py-1 pr-3 " + SENTIMENT_CLASS[deltaSentiment(d.metricName, d.key, d.delta)]}>
                {d.delta !== null ? d.delta.toFixed(3) : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-neutral-500">
        Trajectories that changed verdict ({changes.length})
      </h2>
      <div className="divide-y divide-neutral-200 text-sm dark:divide-neutral-800">
        {changes.map((c, i) => (
          <div key={i} className="py-2">
            <div>{c.question}</div>
            <div className="font-mono text-xs text-neutral-500">
              {c.metric}: {fmtValue(c.from)} &rarr; {fmtValue(c.to)}
            </div>
          </div>
        ))}
        {changes.length === 0 && (
          <p className="py-4 text-neutral-500">No verdicts changed between these two runs.</p>
        )}
      </div>
    </div>
  );
}
