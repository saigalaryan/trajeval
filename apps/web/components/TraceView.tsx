"use client";

import { useMemo, useState } from "react";
import type { Step, TrajectoryResult } from "@/lib/types";
import { fmtValue } from "@/lib/format";

const STEP_BORDER: Record<Step["step_type"], string> = {
  thought: "border-neutral-300 dark:border-neutral-700",
  retrieval: "border-blue-400",
  tool: "border-amber-500",
  answer: "border-emerald-600",
};

// Trajectories with many tool/retrieval steps (each with its own chunk
// list) can render as a very long scroll; steps beyond this are hidden
// behind a "show all" button by default so opening a trace doesn't dump a
// huge DOM tree immediately.
const DEFAULT_VISIBLE_STEPS = 15;

function stepSearchText(step: Step): string {
  switch (step.step_type) {
    case "thought":
      return step.text;
    case "answer":
      return step.text;
    case "tool":
      return `${step.tool_name} ${JSON.stringify(step.args)} ${String(step.result)}`;
    case "retrieval":
      return `${step.query} ${step.chunks.map((c) => `${c.doc_id} ${c.text}`).join(" ")}`;
  }
}

function StepView({
  step,
  index,
  collapsed,
  onToggle,
}: {
  step: Step;
  index: number;
  collapsed: boolean;
  onToggle: () => void;
}) {
  return (
    <div className={`border-l-2 pl-3 py-1 font-mono text-xs ${STEP_BORDER[step.step_type]}`}>
      <button
        type="button"
        onClick={onToggle}
        className="mr-2 text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-300"
        aria-label={collapsed ? "expand step" : "collapse step"}
      >
        {collapsed ? "▸" : "▾"}
      </button>
      <span className="mr-2 text-neutral-400">[{index}]</span>
      {collapsed ? (
        <span className="text-neutral-500">
          {step.step_type}
          {step.step_type === "retrieval" && ` (${step.chunks.length} chunk(s))`}
        </span>
      ) : (
        <>
          {step.step_type === "thought" && <span>thought: {step.text}</span>}
          {step.step_type === "retrieval" && (
            <div className="inline">
              <span>
                retrieval: <span className="font-semibold">{step.query}</span>
              </span>
              <ul className="ml-4 mt-1 list-disc space-y-0.5 text-neutral-500">
                {step.chunks.length === 0 && <li className="italic">no chunks</li>}
                {step.chunks.map((c, i) => (
                  <li key={i}>
                    <span className="font-semibold">{c.doc_id}</span>
                    {c.score !== null && <span> (score={c.score.toFixed(3)})</span>}:{" "}
                    {c.text.slice(0, 200)}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {step.step_type === "tool" && (
            <span>
              tool: <span className="font-semibold">{step.tool_name}</span>(
              {JSON.stringify(step.args)}) &rarr; {fmtValue(step.result)}
            </span>
          )}
          {step.step_type === "answer" && <span>answer: {step.text}</span>}
        </>
      )}
    </div>
  );
}

export function TraceView({ tr }: { tr: TrajectoryResult }) {
  const [search, setSearch] = useState("");
  const [collapsedSteps, setCollapsedSteps] = useState<Set<number>>(new Set());
  const [showAll, setShowAll] = useState(false);

  const toggleStep = (index: number) => {
    setCollapsedSteps((cur) => {
      const next = new Set(cur);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  const steps = useMemo(() => tr.trajectory?.steps ?? [], [tr.trajectory]);

  const matchedIndices = useMemo(() => {
    if (!search.trim()) return null;
    const q = search.trim().toLowerCase();
    return new Set(
      steps
        .map((step, i) => (stepSearchText(step).toLowerCase().includes(q) ? i : -1))
        .filter((i) => i !== -1)
    );
  }, [steps, search]);

  if (tr.error) {
    return <div className="font-mono text-xs text-red-600 dark:text-red-400">Error: {tr.error}</div>;
  }
  if (!tr.trajectory) {
    return <div className="text-xs text-neutral-500">No trajectory recorded.</div>;
  }

  // Search overrides the show-all truncation — if you're looking for
  // something, you want every matching step, not just the first N.
  const visibleIndices = steps
    .map((_, i) => i)
    .filter((i) => matchedIndices === null || matchedIndices.has(i));
  const truncate = matchedIndices === null && !showAll && visibleIndices.length > DEFAULT_VISIBLE_STEPS;
  const shown = truncate ? visibleIndices.slice(0, DEFAULT_VISIBLE_STEPS) : visibleIndices;

  return (
    <div className="space-y-1">
      <div className="mb-2 flex items-center gap-2">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder={`search ${steps.length} step(s)...`}
          className="w-full max-w-xs rounded border border-neutral-300 bg-transparent px-1.5 py-0.5 font-mono text-xs dark:border-neutral-700"
        />
        {matchedIndices !== null && (
          <span className="text-xs text-neutral-500">
            {matchedIndices.size} match{matchedIndices.size === 1 ? "" : "es"}
          </span>
        )}
      </div>

      {shown.length === 0 && (
        <div className="text-xs italic text-neutral-500">No steps match &ldquo;{search}&rdquo;.</div>
      )}
      {shown.map((i) => (
        <StepView
          key={i}
          step={steps[i]}
          index={i}
          collapsed={collapsedSteps.has(i)}
          onToggle={() => toggleStep(i)}
        />
      ))}
      {truncate && (
        <button
          type="button"
          onClick={() => setShowAll(true)}
          className="text-xs text-neutral-500 underline"
        >
          Show all {visibleIndices.length} steps ({visibleIndices.length - DEFAULT_VISIBLE_STEPS} more)
        </button>
      )}

      <div className="mt-2 border-t border-dashed border-neutral-300 pt-2 text-xs dark:border-neutral-700">
        <span className="font-semibold">Final answer: </span>
        {tr.trajectory.final_answer}
      </div>
      <div className="mt-2 flex flex-wrap gap-3 text-xs text-neutral-500">
        {Object.entries(tr.metric_results).map(([name, result]) => (
          <span key={name}>
            {name}: <span className="font-mono">{fmtValue(result.value)}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
