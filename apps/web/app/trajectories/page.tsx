"use client";

import { Suspense, useCallback, useMemo } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useStore } from "@/lib/store";
import { fmtValue } from "@/lib/format";
import { EmptyState } from "@/components/EmptyState";
import { TraceView } from "@/components/TraceView";
import type { TrajectoryResult } from "@/lib/types";

type SortDir = "asc" | "desc";

// Filter/sort/expanded-row state lives in the URL (?tag=&sort=&dir=&fail=&open=)
// instead of useState, so a filtered view is shareable/bookmarkable and
// survives a reload. useSearchParams() requires a <Suspense> boundary in a
// statically-exported app (see app/page.tsx's AutoLoadFromSrc for the same
// pattern) since it reads the client-side URL, not something the static
// export can prerender.
export default function TrajectoryExplorerPage() {
  return (
    <Suspense fallback={null}>
      <TrajectoryExplorer />
    </Suspense>
  );
}

function TrajectoryExplorer() {
  const { primary } = useStore();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const tagFilter = searchParams.get("tag") ?? "__all__";
  const failureOnly = searchParams.get("fail") === "1";
  const sortMetric = searchParams.get("sort") ?? "__none__";
  const sortDir: SortDir = searchParams.get("dir") === "desc" ? "desc" : "asc";
  const expanded = searchParams.get("open");

  // Merges a partial update into the current params and replaces the URL
  // (not push) so filtering doesn't spam browser history one entry per
  // keystroke/click. `null` deletes a key instead of writing it literally,
  // keeping the URL clean of default values.
  const updateParams = useCallback(
    (patch: Record<string, string | null>) => {
      const next = new URLSearchParams(searchParams.toString());
      for (const [key, value] of Object.entries(patch)) {
        if (value === null) next.delete(key);
        else next.set(key, value);
      }
      const qs = next.toString();
      router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
    },
    [pathname, router, searchParams]
  );

  const allTags = useMemo(() => {
    if (!primary) return [];
    const tags = new Set<string>();
    for (const tr of primary.trajectory_results) for (const t of tr.tags) tags.add(t);
    return Array.from(tags).sort();
  }, [primary]);

  const metricNames = primary?.metadata.metric_names ?? [];

  const rows = useMemo(() => {
    if (!primary) return [];
    let list = primary.trajectory_results;
    if (tagFilter !== "__all__") list = list.filter((tr) => tr.tags.includes(tagFilter));
    if (failureOnly) {
      list = list.filter((tr) => {
        if (tr.error) return true;
        // "failure" for a metric = value present and < 1.0 (works for the
        // 0/1 and 0..1 scored metrics; termination's inverted scale means
        // "failure" there reads as "excess steps > 0", which this same
        // check happens to also catch since higher is worse there too).
        return Object.values(tr.metric_results).some((r) => r.value !== null && r.value < 1);
      });
    }
    if (sortMetric !== "__none__") {
      list = [...list].sort((a, b) => {
        const av = a.metric_results[sortMetric]?.value;
        const bv = b.metric_results[sortMetric]?.value;
        const an = av === null || av === undefined ? Infinity : av;
        const bn = bv === null || bv === undefined ? Infinity : bv;
        return sortDir === "asc" ? an - bn : bn - an;
      });
    }
    return list;
  }, [primary, tagFilter, failureOnly, sortMetric, sortDir]);

  if (!primary) return <EmptyState />;

  return (
    <div>
      <h1 className="mb-4 text-lg font-semibold">Trajectories ({rows.length})</h1>

      <div className="mb-4 flex flex-wrap items-center gap-3 text-sm">
        <label className="flex items-center gap-1.5">
          tag:
          <select
            value={tagFilter}
            onChange={(e) =>
              updateParams({ tag: e.target.value === "__all__" ? null : e.target.value })
            }
            className="rounded border border-neutral-300 bg-transparent px-1.5 py-0.5 dark:border-neutral-700"
          >
            <option value="__all__">all</option>
            {allTags.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-1.5">
          sort by:
          <select
            value={sortMetric}
            onChange={(e) =>
              updateParams({ sort: e.target.value === "__none__" ? null : e.target.value })
            }
            className="rounded border border-neutral-300 bg-transparent px-1.5 py-0.5 dark:border-neutral-700"
          >
            <option value="__none__">none</option>
            {metricNames.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>

        {sortMetric !== "__none__" && (
          <button
            type="button"
            onClick={() => updateParams({ dir: sortDir === "asc" ? "desc" : null })}
            className="rounded border border-neutral-300 px-2 py-0.5 dark:border-neutral-700"
          >
            {sortDir === "asc" ? "worst first" : "best first"}
          </button>
        )}

        <label className="flex items-center gap-1.5">
          <input
            type="checkbox"
            checked={failureOnly}
            onChange={(e) => updateParams({ fail: e.target.checked ? "1" : null })}
          />
          failures only
        </label>
      </div>

      <div className="divide-y divide-neutral-200 dark:divide-neutral-800">
        {rows.map((tr) => {
          const rowId = tr.trajectory_id ?? tr.golden_id;
          return (
            <TrajectoryRow
              key={rowId}
              tr={tr}
              expanded={expanded === rowId}
              onToggle={() => updateParams({ open: expanded === rowId ? null : rowId })}
            />
          );
        })}
        {rows.length === 0 && <p className="py-4 text-sm text-neutral-500">No trajectories match.</p>}
      </div>
    </div>
  );
}

function TrajectoryRow({
  tr,
  expanded,
  onToggle,
}: {
  tr: TrajectoryResult;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="py-2">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-start gap-3 text-left"
      >
        <span className="mt-0.5 font-mono text-xs text-neutral-400">{expanded ? "▾" : "▸"}</span>
        <div className="flex-1">
          <div className="text-sm">{tr.question}</div>
          <div className="mt-1 flex flex-wrap gap-2 text-xs text-neutral-500">
            {tr.tags.map((t) => (
              <span key={t} className="rounded bg-neutral-100 px-1.5 py-0.5 dark:bg-neutral-800">
                {t}
              </span>
            ))}
            {tr.error ? (
              <span className="text-red-600 dark:text-red-400">error</span>
            ) : (
              Object.entries(tr.metric_results).map(([name, r]) => (
                <span key={name} className="font-mono">
                  {name}={fmtValue(r.value)}
                </span>
              ))
            )}
          </div>
        </div>
      </button>
      {expanded && (
        <div className="mt-2 ml-6 rounded border border-neutral-200 p-3 dark:border-neutral-800">
          <TraceView tr={tr} />
        </div>
      )}
    </div>
  );
}
