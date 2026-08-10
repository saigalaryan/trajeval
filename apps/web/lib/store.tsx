"use client";

// Client-only in-memory store for loaded RunResults. No backend, no
// database — everything lives in the browser tab. Persists to
// sessionStorage so a reload during the same tab session doesn't lose the
// loaded run(s); closing the tab clears it, which is the right default for
// data that may include a customer's actual questions/answers.

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { isRunResult, type RunResult } from "./types";

export interface TrendRun {
  result: RunResult;
  name: string;
}

interface StoreState {
  primary: RunResult | null;
  primaryName: string | null;
  secondary: RunResult | null;
  secondaryName: string | null;
  // Runs loaded for the /trend page — an open-ended list, unlike
  // primary/secondary's fixed two slots, since a trend is only meaningful
  // across more than two points.
  trend: TrendRun[];
  loadPrimary: (result: RunResult, name: string) => void;
  loadSecondary: (result: RunResult, name: string) => void;
  addTrendRun: (result: RunResult, name: string) => void;
  removeTrendRun: (runId: string) => void;
  clearTrend: () => void;
  clear: () => void;
}

const StoreContext = createContext<StoreState | null>(null);

const STORAGE_KEY = "trajeval.store.v1";

export function StoreProvider({ children }: { children: ReactNode }) {
  const [primary, setPrimary] = useState<RunResult | null>(null);
  const [primaryName, setPrimaryName] = useState<string | null>(null);
  const [secondary, setSecondary] = useState<RunResult | null>(null);
  const [secondaryName, setSecondaryName] = useState<string | null>(null);
  const [trend, setTrend] = useState<TrendRun[]>([]);

  // Deliberate one-time post-mount hydration from sessionStorage, not a
  // candidate for a lazy useState initializer: this is a static-exported
  // app, so the first render (both the prerendered HTML and the client's
  // initial hydration pass) must happen with `window`/`sessionStorage`
  // unavailable — reading it in the initializer would throw during export
  // and would mismatch the prerendered markup on hydration either way. The
  // effect's setState calls below intentionally trigger one extra render
  // after mount to bring in whatever was saved in a prior session.
  useEffect(() => {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    try {
      const parsed = JSON.parse(raw);
      if (parsed.primary && isRunResult(parsed.primary)) {
        // eslint-disable-next-line react-hooks/set-state-in-effect -- see comment above
        setPrimary(parsed.primary);
        setPrimaryName(parsed.primaryName ?? "run");
      }
      if (parsed.secondary && isRunResult(parsed.secondary)) {
        setSecondary(parsed.secondary);
        setSecondaryName(parsed.secondaryName ?? "run");
      }
      if (Array.isArray(parsed.trend)) {
        const restored: TrendRun[] = parsed.trend.filter(
          (t: unknown): t is TrendRun =>
            typeof t === "object" &&
            t !== null &&
            isRunResult((t as { result?: unknown }).result) &&
            typeof (t as { name?: unknown }).name === "string"
        );
        setTrend(restored);
      }
    } catch {
      // corrupted session storage — ignore and start fresh
    }
  }, []);

  const persist = useCallback(
    (next: {
      primary: RunResult | null;
      primaryName: string | null;
      secondary: RunResult | null;
      secondaryName: string | null;
      trend: TrendRun[];
    }) => {
      try {
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        // storage quota exceeded on a very large run (trend runs multiply
        // this risk — N runs, not one) — non-fatal, just means a reload
        // won't restore it
      }
    },
    []
  );

  const loadPrimary = useCallback(
    (result: RunResult, name: string) => {
      setPrimary(result);
      setPrimaryName(name);
      persist({ primary: result, primaryName: name, secondary, secondaryName, trend });
    },
    [persist, secondary, secondaryName, trend]
  );

  const loadSecondary = useCallback(
    (result: RunResult, name: string) => {
      setSecondary(result);
      setSecondaryName(name);
      persist({ primary, primaryName, secondary: result, secondaryName: name, trend });
    },
    [persist, primary, primaryName, trend]
  );

  const addTrendRun = useCallback(
    (result: RunResult, name: string) => {
      // Re-adding the same run (same file dropped twice, or a page reload's
      // hydration racing a fresh add) replaces rather than duplicates it —
      // keyed on run_id, which is stable per real run.
      const runId = result.metadata.run_id;
      const next = [...trend.filter((t) => t.result.metadata.run_id !== runId), { result, name }];
      setTrend(next);
      persist({ primary, primaryName, secondary, secondaryName, trend: next });
    },
    [persist, primary, primaryName, secondary, secondaryName, trend]
  );

  const removeTrendRun = useCallback(
    (runId: string) => {
      const next = trend.filter((t) => t.result.metadata.run_id !== runId);
      setTrend(next);
      persist({ primary, primaryName, secondary, secondaryName, trend: next });
    },
    [persist, primary, primaryName, secondary, secondaryName, trend]
  );

  const clearTrend = useCallback(() => {
    setTrend([]);
    persist({ primary, primaryName, secondary, secondaryName, trend: [] });
  }, [persist, primary, primaryName, secondary, secondaryName]);

  const clear = useCallback(() => {
    setPrimary(null);
    setPrimaryName(null);
    setSecondary(null);
    setSecondaryName(null);
    setTrend([]);
    sessionStorage.removeItem(STORAGE_KEY);
  }, []);

  const value = useMemo(
    () => ({
      primary,
      primaryName,
      secondary,
      secondaryName,
      trend,
      loadPrimary,
      loadSecondary,
      addTrendRun,
      removeTrendRun,
      clearTrend,
      clear,
    }),
    [
      primary,
      primaryName,
      secondary,
      secondaryName,
      trend,
      loadPrimary,
      loadSecondary,
      addTrendRun,
      removeTrendRun,
      clearTrend,
      clear,
    ]
  );

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

export function useStore(): StoreState {
  const ctx = useContext(StoreContext);
  if (!ctx) throw new Error("useStore must be used within StoreProvider");
  return ctx;
}
