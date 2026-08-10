"use client";

// Loads a RunResult JSON entirely client-side — no backend involved. Three
// ways in: drag-and-drop a file, use the file picker, or pass a `?src=`
// query param pointing at a URL trajeval serves some other way (e.g.
// `python -m http.server` in the results directory). All three work
// identically in a static export.

import { useCallback, useRef, useState } from "react";
import { isRunResult, type RunResult } from "@/lib/types";

interface FileLoaderProps {
  label: string;
  onLoaded: (result: RunResult, name: string) => void;
  // Accept and process every dropped/picked file instead of just the
  // first — used by the /trend page, where the point is loading several
  // runs at once rather than replacing a single slot.
  multiple?: boolean;
}

export function FileLoader({ label, onLoaded, multiple = false }: FileLoaderProps) {
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      try {
        const text = await file.text();
        const parsed = JSON.parse(text);
        if (!isRunResult(parsed)) {
          setError(`${file.name} doesn't look like a trajeval RunResult JSON file.`);
          return;
        }
        onLoaded(parsed, file.name);
      } catch (e) {
        setError(`Couldn't parse ${file.name}: ${e instanceof Error ? e.message : String(e)}`);
      }
    },
    [onLoaded]
  );

  const handleFiles = useCallback(
    async (files: FileList) => {
      setError(null);
      const list = multiple ? Array.from(files) : files[0] ? [files[0]] : [];
      for (const file of list) {
        await handleFile(file);
      }
    },
    [handleFile, multiple]
  );

  return (
    <div
      className={`rounded border-2 border-dashed p-6 text-center transition-colors ${
        dragging
          ? "border-neutral-500 bg-neutral-100 dark:bg-neutral-800"
          : "border-neutral-300 dark:border-neutral-700"
      }`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (e.dataTransfer.files.length > 0) void handleFiles(e.dataTransfer.files);
      }}
    >
      <p className="mb-2 font-mono text-sm text-neutral-600 dark:text-neutral-400">{label}</p>
      <p className="mb-3 text-sm text-neutral-500">
        Drag {multiple ? "RunResult JSON files" : "a RunResult JSON file"} here, or
      </p>
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        className="rounded border border-neutral-400 px-3 py-1.5 text-sm font-medium hover:bg-neutral-100 dark:border-neutral-600 dark:hover:bg-neutral-800"
      >
        Choose file{multiple ? "s" : ""}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="application/json,.json"
        multiple={multiple}
        className="hidden"
        onChange={(e) => {
          if (e.target.files && e.target.files.length > 0) void handleFiles(e.target.files);
          e.target.value = "";
        }}
      />
      {error && <p className="mt-3 text-sm text-red-600 dark:text-red-400">{error}</p>}
    </div>
  );
}
