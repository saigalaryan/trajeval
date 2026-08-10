"use client"; // Error boundaries must be Client Components

// Catches crashes in the root layout itself (StoreProvider, NavBar) —
// app/error.tsx can't catch these since it's rendered *inside* the layout.
// Rare in practice (the layout does little besides read sessionStorage
// defensively — see lib/store.tsx), but a crash here would otherwise be a
// blank white page with nothing but a browser console error, which is
// worse than this project's "no error boundary" gap for page-level crashes.
// Must define its own <html>/<body> — it replaces the root layout when active.
export default function GlobalError({
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  return (
    <html lang="en" className="h-full">
      <body className="flex h-full min-h-full items-center justify-center bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100">
        <div className="mx-auto max-w-md text-center">
          <h2 className="mb-2 text-sm font-semibold">Something went wrong.</h2>
          <p className="mb-4 text-xs text-neutral-500">
            trajeval failed to load. Reloading may help.
          </p>
          <button type="button" onClick={() => retry()} className="text-sm underline">
            Try again
          </button>
        </div>
      </body>
    </html>
  );
}
