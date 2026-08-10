"use client";

import { Suspense, useMemo } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useStore, type TrendRun } from "@/lib/store";
import { FileLoader } from "@/components/FileLoader";
import { fmtValue } from "@/lib/format";

// Tracks one aggregate metric/key across several loaded RunResults, sorted
// by RunMetadata.started_at — the multi-run counterpart to /compare's
// two-run diff. Doesn't require a `primary` run to be loaded (unlike every
// other page): the point here is a set of N runs, and `primary` is just
// "the one you're currently focused on" for the rest of the app.
export default function TrendPage() {
  return (
    <Suspense fallback={null}>
      <Trend />
    </Suspense>
  );
}

function sortedTrend(trend: TrendRun[]): TrendRun[] {
  return [...trend].sort(
    (a, b) =>
      new Date(a.result.metadata.started_at).getTime() -
      new Date(b.result.metadata.started_at).getTime()
  );
}

function Trend() {
  const { trend, addTrendRun, removeTrendRun, clearTrend } = useStore();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const runs = useMemo(() => sortedTrend(trend), [trend]);

  const metricNames = useMemo(() => {
    const names = new Set<string>();
    for (const r of runs) for (const name of Object.keys(r.result.aggregate_scores)) names.add(name);
    return Array.from(names).sort();
  }, [runs]);

  const selectedMetric = searchParams.get("metric") ?? metricNames[0] ?? null;

  const keys = useMemo(() => {
    if (!selectedMetric) return [];
    const ks = new Set<string>();
    for (const r of runs) {
      const agg = r.result.aggregate_scores[selectedMetric];
      if (!agg) continue;
      for (const [k, v] of Object.entries(agg)) if (typeof v === "number") ks.add(k);
    }
    return Array.from(ks).sort();
  }, [runs, selectedMetric]);

  const selectedKey = searchParams.get("key") ?? keys[0] ?? null;

  function updateParams(patch: Record<string, string | null>) {
    const next = new URLSearchParams(searchParams.toString());
    for (const [k, v] of Object.entries(patch)) {
      if (v === null) next.delete(k);
      else next.set(k, v);
    }
    const qs = next.toString();
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  }

  const points: TrendPoint[] = useMemo(() => {
    if (!selectedMetric || !selectedKey) return [];
    return runs.map((r) => {
      const raw = r.result.aggregate_scores[selectedMetric]?.[selectedKey];
      return {
        runId: r.result.metadata.run_id,
        name: r.name,
        startedAt: r.result.metadata.started_at,
        value: typeof raw === "number" ? raw : null,
      };
    });
  }, [runs, selectedMetric, selectedKey]);

  return (
    <div>
      <h1 className="mb-1 text-lg font-semibold">Trend</h1>
      <p className="mb-4 max-w-xl text-sm text-neutral-500">
        Load several RunResults — e.g. saved CI history — to track one aggregate metric across
        runs over time, sorted by when each run started.
      </p>

      <FileLoader label="Add run(s) to trend" multiple onLoaded={addTrendRun} />

      {runs.length === 0 ? (
        <p className="mt-6 text-sm text-neutral-500">No runs added yet.</p>
      ) : (
        <>
          <div className="my-4 flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-3 text-sm">
              <label className="flex items-center gap-1.5">
                metric:
                <select
                  value={selectedMetric ?? ""}
                  onChange={(e) => updateParams({ metric: e.target.value, key: null })}
                  className="rounded border border-neutral-300 bg-transparent px-1.5 py-0.5 dark:border-neutral-700"
                >
                  {metricNames.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex items-center gap-1.5">
                key:
                <select
                  value={selectedKey ?? ""}
                  onChange={(e) => updateParams({ key: e.target.value })}
                  className="rounded border border-neutral-300 bg-transparent px-1.5 py-0.5 dark:border-neutral-700"
                >
                  {keys.map((k) => (
                    <option key={k} value={k}>
                      {k}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <button
              type="button"
              onClick={clearTrend}
              className="text-xs text-neutral-500 underline"
            >
              clear all ({runs.length})
            </button>
          </div>

          <TrendChart points={points} />

          <table className="mt-6 w-full max-w-2xl border-collapse text-sm">
            <thead>
              <tr className="border-b border-neutral-200 text-left text-xs text-neutral-500 dark:border-neutral-800">
                <th className="py-1 pr-3 font-normal">#</th>
                <th className="py-1 pr-3 font-normal">Run</th>
                <th className="py-1 pr-3 font-normal">Started</th>
                <th className="py-1 pr-3 font-normal">Value</th>
                <th className="py-1 pr-3 font-normal" />
              </tr>
            </thead>
            <tbody>
              {runs.map((r, i) => (
                <tr
                  key={r.result.metadata.run_id}
                  className="border-b border-neutral-100 font-mono dark:border-neutral-900"
                >
                  <td className="py-1 pr-3 text-neutral-500">{i + 1}</td>
                  <td className="py-1 pr-3">{r.name}</td>
                  <td className="py-1 pr-3 text-neutral-500">{r.result.metadata.started_at}</td>
                  <td className="py-1 pr-3">
                    {selectedMetric && selectedKey
                      ? fmtValue(r.result.aggregate_scores[selectedMetric]?.[selectedKey])
                      : "—"}
                  </td>
                  <td className="py-1 pr-3">
                    <button
                      type="button"
                      onClick={() => removeTrendRun(r.result.metadata.run_id)}
                      className="text-xs text-neutral-500 hover:text-red-600 dark:hover:text-red-400"
                    >
                      remove
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

interface TrendPoint {
  runId: string;
  name: string;
  startedAt: string;
  value: number | null;
}

function TrendChart({ points }: { points: TrendPoint[] }) {
  const width = 640;
  const height = 220;
  const padding = { top: 16, right: 16, bottom: 28, left: 48 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const withValues = points
    .map((p, i) => ({ ...p, index: i }))
    .filter((p): p is TrendPoint & { index: number; value: number } => p.value !== null);

  if (withValues.length === 0) {
    return (
      <p className="text-sm text-neutral-500">
        None of the loaded runs have a numeric value for this metric/key.
      </p>
    );
  }

  const values = withValues.map((p) => p.value);
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const yPad = (max - min) * 0.1;
  min -= yPad;
  max += yPad;

  const xFor = (i: number) => (points.length === 1 ? plotW / 2 : (i / (points.length - 1)) * plotW);
  const yFor = (v: number) => plotH - ((v - min) / (max - min)) * plotH;

  const pathD = withValues.map((p) => `${xFor(p.index)},${yFor(p.value)}`).join(" L ");

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full max-w-2xl text-neutral-400"
      role="img"
      aria-label="Metric value across runs"
    >
      <g transform={`translate(${padding.left},${padding.top})`}>
        {[0, 0.5, 1].map((t) => {
          const v = min + t * (max - min);
          const y = plotH - t * plotH;
          return (
            <g key={t}>
              <line x1={0} x2={plotW} y1={y} y2={y} stroke="currentColor" strokeOpacity={0.15} />
              <text
                x={-6}
                y={y}
                textAnchor="end"
                dominantBaseline="middle"
                fontSize={10}
                fill="currentColor"
              >
                {v.toFixed(2)}
              </text>
            </g>
          );
        })}

        {withValues.length > 1 && (
          <path
            d={`M ${pathD}`}
            fill="none"
            stroke="currentColor"
            className="text-blue-500"
            strokeWidth={2}
          />
        )}

        {withValues.map((p) => (
          <circle key={p.runId} cx={xFor(p.index)} cy={yFor(p.value)} r={3} className="fill-blue-500">
            <title>{`${p.name}: ${p.value}`}</title>
          </circle>
        ))}

        {points.map((p, i) => (
          <text key={p.runId} x={xFor(i)} y={plotH + 16} textAnchor="middle" fontSize={9} fill="currentColor">
            {i + 1}
          </text>
        ))}
      </g>
    </svg>
  );
}
