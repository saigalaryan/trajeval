# Changelog

All notable changes to `trajeval` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project has
not yet made a versioned release, so everything so far lives under
`[Unreleased]`.

## [Unreleased]

### Added

- `trajeval serve [results_dir]` — serves the bundled web viewer locally
  against a results directory, with no Node.js dependency for the end user.
  The viewer is built once at release time (`scripts/bundle_webapp.py`) and
  shipped as static files inside the wheel.
- Weighted Cohen's kappa (`weights="linear"` / `"quadratic"`) for
  judge/human calibration agreement on ordinal metrics. `query_quality`
  (rated 1–5) now uses quadratic weighting, so a judge/human near-miss
  ("3 vs 4") is penalized less than a far miss ("1 vs 5") — unweighted kappa
  treated both the same.
- Per-metric scoring failure isolation: if one judged metric throws (e.g. a
  malformed judge response) while scoring a trajectory, only that
  metric/trajectory pair is marked failed (`TrajectoryResult.metric_errors`)
  — the rest of the run completes instead of the whole run aborting.
- Web viewer: error boundaries (`app/error.tsx`, `app/global-error.tsx`) for
  rendering crashes; `?src=<url>` query param to auto-load a `RunResult`
  from a URL on the loader page; filter/sort/expanded-row state on
  `/trajectories` now lives in the URL (shareable, survives reload);
  trace view gained a step search box and per-step / truncated-list collapse
  for long trajectories.
- `AnthropicJudgeClient` test suite against a fake `anthropic.Anthropic` SDK
  (prompt/system construction, retries, cache hit/miss, cost tracking).
- `OpenAIJudgeClient` — a second `JudgeClient` implementation backed by the
  OpenAI API, same constructor shape as `AnthropicJudgeClient`. Select it
  per-config with `judge_provider: openai` (default remains `anthropic`);
  `trajeval.cost`'s pricing table now covers a handful of OpenAI models too.
- `trajeval validate <dataset>` — lints a golden dataset JSONL before you
  spend time (or, with judged metrics, money) running against it: malformed
  lines, missing/invalid fields, and duplicate ids, all reported at once
  instead of one-at-a-time on the first `trajeval run`.
- `trajeval run` now shows a live progress bar (trajectories scored / total,
  elapsed time) instead of blocking silently — `runner.run()` grew an
  `on_progress` callback that the CLI wires to `rich`.
- Web viewer: a `/trend` page tracking one aggregate metric across any
  number of loaded runs (sorted by `started_at`), for e.g. watching a CI
  history over time — the multi-run counterpart to the two-run `/compare`.
- A CI workflow (`.github/workflows/ci.yml`) now runs `ruff`, `mypy --strict`,
  `pytest --cov`, and an `mkdocs build --strict` check for the Python
  package, and `lint`/`tsc --noEmit`/`build` for the web viewer, on every
  push and PR to this repo.

### Fixed

- The judge response cache (`.trajeval_judge_cache.json`) is now gitignored
  by default — it can contain plaintext prompts/responses derived from
  whatever was in your dataset.
- Compare page delta coloring in the web viewer respected only a hardcoded
  "higher is better" assumption, so `termination`'s lower-is-better metrics
  showed improvements in red and regressions in green. Delta sentiment is
  now computed per-metric-key (`apps/web/lib/metricDirection.ts`).
- `trajeval serve`'s clean-URL routing (`/run` → `run.html`) could resolve
  to the wrong file: Next.js's App Router static export emits a same-named
  *directory* of RSC payload chunks alongside `run.html`, and the original
  existence check matched that directory before ever falling through to the
  `.html` file.

### Docs

- `CONTRIBUTING.md`, this changelog, and an MkDocs + mkdocstrings API
  reference (`packages/trajeval/docs/`, `mkdocs.yml`) were added. A `LICENSE`
  (MIT) file now exists at the repo root — the README referenced this
  license from the start, but no file backed it.
