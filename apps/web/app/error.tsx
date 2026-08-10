"use client"; // Error boundaries must be Client Components

import { useEffect } from "react";
import Link from "next/link";

// Catches rendering crashes anywhere under app/ — a malformed-but-not-quite
// rejected RunResult reaching a page that assumes a field exists, a bad
// index into trajectory_results, etc. isRunResult() in lib/types.ts filters
// out the obvious shape mismatches before anything reaches the store, but
// it's a shallow check; this is the backstop for whatever gets through it
// or fails for some other reason (browser storage quirks, a future schema
// field a page isn't ready for). Note the `retry` prop name, not `reset` —
// this Next.js version renamed it (see AGENTS.md).
export default function ErrorBoundary({
  error,
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="mx-auto max-w-md text-center">
      <h2 className="mb-2 text-sm font-semibold">Something went wrong.</h2>
      <p className="mb-4 text-xs text-neutral-500">
        {error.message || "The page crashed while rendering this run."}
      </p>
      <div className="flex justify-center gap-4 text-sm">
        <button type="button" onClick={() => retry()} className="underline">
          Try again
        </button>
        <Link href="/" className="underline">
          Load a different run
        </Link>
      </div>
    </div>
  );
}
