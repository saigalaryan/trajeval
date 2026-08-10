"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { FileLoader } from "@/components/FileLoader";
import { useStore } from "@/lib/store";
import { isRunResult, type RunResult } from "@/lib/types";

/** Auto-loads `?src=<url>` on mount — this is what makes `trajeval serve`
 * (or anyone hosting a RunResult next to this static export) a one-click
 * open instead of a manual file-picker step. Wrapped in Suspense per
 * Next.js's requirement for `useSearchParams` in a statically exported page.
 */
function AutoLoadFromSrc() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { loadPrimary } = useStore();
  const [error, setError] = useState<string | null>(null);
  const src = searchParams.get("src");

  useEffect(() => {
    if (!src) return;
    let cancelled = false;

    fetch(src)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (cancelled) return;
        if (!isRunResult(data)) {
          setError(`${src} doesn't look like a trajeval RunResult JSON file.`);
          return;
        }
        loadPrimary(data, src.split("/").pop() ?? src);
        router.replace("/run");
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(`Couldn't load ${src}: ${e instanceof Error ? e.message : String(e)}`);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [src, loadPrimary, router]);

  if (!src) return null;
  if (error) {
    return <p className="mb-4 text-sm text-red-600 dark:text-red-400">{error}</p>;
  }
  return <p className="mb-4 text-sm text-neutral-500">Loading {src}…</p>;
}

export default function HomePage() {
  const router = useRouter();
  const { primary, loadPrimary } = useStore();

  function handleLoaded(result: RunResult, name: string) {
    loadPrimary(result, name);
    router.push("/run");
  }

  return (
    <div className="mx-auto max-w-xl">
      <h1 className="mb-1 text-xl font-semibold">trajeval</h1>
      <p className="mb-6 text-sm text-neutral-500">
        A viewer for trajeval RunResult JSON files. Nothing is uploaded anywhere — the file is
        parsed entirely in your browser.
      </p>

      <Suspense fallback={null}>
        <AutoLoadFromSrc />
      </Suspense>

      <FileLoader label="Load a RunResult JSON" onLoaded={handleLoaded} />

      {primary && (
        <p className="mt-4 text-sm">
          A run is already loaded —{" "}
          <a href="/run" className="underline">
            go to the run overview
          </a>
          , or load a different file above to replace it.
        </p>
      )}

      <p className="mt-8 text-xs text-neutral-500">
        Generate a RunResult with{" "}
        <code className="rounded bg-neutral-100 px-1 py-0.5 dark:bg-neutral-800">
          trajeval run --config trajeval.yaml --out results/latest.json
        </code>
        . Serve this viewer pointed at a results directory with{" "}
        <code className="rounded bg-neutral-100 px-1 py-0.5 dark:bg-neutral-800">
          trajeval serve results/
        </code>
        , or pass <code className="rounded bg-neutral-100 px-1 py-0.5 dark:bg-neutral-800">?src=</code>{" "}
        yourself to auto-load a specific file.
      </p>
    </div>
  );
}
