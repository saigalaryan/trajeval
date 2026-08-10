# Contributing to trajeval

Thanks for looking at this. This is a small monorepo — `packages/trajeval`
(the Python evaluation library + CLI) and `apps/web` (the Next.js results
viewer) — with no database, no hosted backend, and no CI-side secrets
required to run the test suite. Everything below can be verified on your own
machine before opening a PR.

## Repo layout

```
packages/trajeval/   Python library + CLI (uv-managed, Python 3.11+)
apps/web/             Next.js viewer, statically exported (no Node needed to use trajeval — only to build the viewer)
datasets/seed/         A small example golden dataset used by the quickstart
scripts/                One-off maintenance scripts (e.g. bundling apps/web into the wheel)
.github/                Composite action + example workflow for running trajeval in CI
```

The core library (`trajeval.metrics`, `trajeval.runner`, `trajeval.adapters`)
never imports an agent framework (LangGraph, LlamaIndex, etc.) — adapters are
the only place framework-specific code is allowed to live, and they're
optional. Keep that boundary when adding features.

## Setting up

### Python (`packages/trajeval`)

```bash
cd packages/trajeval
uv sync --all-extras   # installs dev, cli, and judge extras
```

Run these before every PR — all four must pass clean:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy --strict src/trajeval
uv run pytest --cov=trajeval --cov-report=term-missing
```

`ruff format .` (without `--check`) and `ruff check --fix .` will fix most
formatting/lint issues for you.

### Web app (`apps/web`)

Only needed if you're touching the viewer:

```bash
cd apps/web
npm install
npm run lint
npx tsc --noEmit
npm run build   # static export to apps/web/out/ — must succeed, it's what ships in the wheel
```

If you change `apps/web` and want `trajeval serve` to reflect it locally,
rebundle it into the Python package:

```bash
python scripts/bundle_webapp.py
```

This is a manual step, not a build hook on `packages/trajeval` — installing
`trajeval` from PyPI must never require Node.js. The bundled output
(`packages/trajeval/src/trajeval/_webapp_dist/`) is gitignored; only a
release build regenerates and ships it.

## Testing conventions

- **Write tests alongside the code that needs them, not after.** A PR that
  adds behavior without a test covering it will get asked for one.
- Prefer a fake over a mock where the collaborator is simple enough to fake
  (see `tests/test_anthropic_judge_client.py`'s fake SDK classes, or
  `FakeJudgeClient` in `tests/test_runner.py`) — real assertions on real
  inputs/outputs, not assertions about which methods got called.
- When a test fails, treat that as a real bug to fix, not a signal to loosen
  the assertion — unless you can point to what's actually wrong with the
  assertion itself.
- Every LLM-judged metric (`query_quality`, `recovery`, `faithfulness`) must
  stay calibratable against human labels with a reportable agreement score
  (see `trajeval.calibration`). If you add a judged metric, wire it into
  calibration the same way.

## Code style

- Boring and readable over clever. Someone who isn't you will read this next.
- `mypy --strict` passes on the whole package — no `# type: ignore` without a
  comment explaining why it's needed.
- Deterministic-before-LLM-judged: if a metric can be computed without a
  judge call, it should be (see `trajeval.metrics.termination`,
  `trajectory_efficiency`, `retrieval_necessity` for the deterministic ones).
  Reach for a judge only when there's genuinely no rule-based alternative.

## Architectural changes

If your change touches the on-disk `RunResult`/`TrajectoryResult` schema,
the CLI's command surface, the adapter interface, or how the web app and
library agree on shapes (`apps/web/lib/types.ts` mirrors
`trajeval.results`/`trajeval.types` by hand — there's no codegen link
between them), open an issue or draft PR describing the change before
writing the implementation. These are the parts of the project other
people's code depends on.

## Sending a PR

1. Branch off `main`.
2. Keep the four Python checks and (if `apps/web` changed) the three web
   checks above green.
3. Update `packages/trajeval/CHANGELOG.md` under `[Unreleased]` if the
   change is user-visible (new metric, CLI flag, bug fix, breaking change).
4. Describe what changed and why, not just what — the "why" is what saves
   the next person from re-deriving your reasoning.

## Reporting bugs / security issues

Open an issue with a minimal repro (a golden dataset entry + adapter output
that triggers it, where possible). This project has no hosted service and
holds no user data beyond whatever you point it at locally, so there's no
separate security-disclosure process — a regular issue is fine.
