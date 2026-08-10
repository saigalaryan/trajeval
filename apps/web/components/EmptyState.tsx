import Link from "next/link";

export function EmptyState() {
  return (
    <div className="mx-auto max-w-md text-center">
      <p className="mb-3 text-sm text-neutral-500">No run loaded yet.</p>
      <Link href="/" className="text-sm underline">
        Load a RunResult JSON
      </Link>
    </div>
  );
}
