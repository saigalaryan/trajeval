# trajeval web viewer

A viewer for `RunResult` JSON files produced by `trajeval run`. Reads JSON,
renders it. **No backend, no database, no auth** — every page is a client
component; the file you load never leaves your browser tab.

If this app disappeared, `trajeval` would still be fully useful — the CLI's
`trajeval report` command already produces a self-contained HTML report.
This app exists for the interactive views a static report can't do: sorting
and filtering the trajectory list, and a live two-run comparison.

## Two ways to use it

**Static export** (the default `npm run build` output, in `out/`) — a plain
folder of HTML/JS/CSS. Serve it with anything (`npx serve out`, S3 + a CDN,
a GitHub Pages deploy) or open `out/index.html` directly. Drag a RunResult
JSON onto the page, or use the file picker.

**Local** — same static export, served locally next to a results directory
so you're not retyping file paths:

```bash
npm run build
npx serve out
```

There's no dedicated `trajeval serve` command yet — that's tracked as
follow-up work, not implemented. For now, build once and serve the `out/`
folder with any static file server.

## Views

- **`/run`** — aggregate scores per metric, calibration badges (uncalibrated
  judged metrics are flagged, never rendered as a clean trustworthy number),
  and per-run metadata.
- **`/trajectories`** — filter by tag, sort by any metric's value, toggle
  "failures only", and expand any row for its full step-by-step trace
  (query issued, chunks returned, tool calls, final answer) inline.
- **`/compare`** — load a second RunResult and see aggregate deltas plus the
  specific trajectories (matched by golden record id) whose pass/fail
  verdict flipped between the two runs.

## Development

```bash
npm install
npm run dev      # http://localhost:3000
npm run build    # static export to out/
npm run lint
```
