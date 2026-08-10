import type { CalibrationState } from "@/lib/types";

export function CalibrationBadge({ state }: { state: CalibrationState | undefined }) {
  if (!state || !state.is_calibrated) {
    return (
      <span className="rounded bg-amber-100 px-1.5 py-0.5 text-xs font-medium text-amber-800 dark:bg-amber-950 dark:text-amber-300">
        uncalibrated
      </span>
    );
  }
  return (
    <span className="rounded bg-emerald-100 px-1.5 py-0.5 text-xs font-medium text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300">
      κ={state.kappa?.toFixed(2)} (n={state.n_labels})
    </span>
  );
}
