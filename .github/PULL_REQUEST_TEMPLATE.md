## What changed and why

<!-- The "why" matters more than the "what" — it's what saves the next
person from re-deriving your reasoning. See CONTRIBUTING.md. -->

## Checklist

- [ ] Tests added/updated alongside the code (not after) — see
      `packages/trajeval/CONTRIBUTING.md`'s testing conventions
- [ ] `uv run ruff format --check .` / `uv run ruff check .` / `uv run mypy --strict src/trajeval` all pass (`packages/trajeval/`)
- [ ] `uv run pytest --cov=trajeval` passes
- [ ] If `apps/web` changed: `npm run lint`, `npx tsc --noEmit`, `npm run build` all pass
- [ ] `packages/trajeval/CHANGELOG.md` updated under `[Unreleased]` if this is user-visible
- [ ] If this touches the `RunResult`/`TrajectoryResult` schema, the CLI
      surface, or the adapter interface: this was discussed in an issue
      first, per CONTRIBUTING.md's architectural-changes note

## Anything else reviewers should know

<!-- Design tradeoffs, things you're unsure about, follow-ups you're
deliberately not doing in this PR, etc. -->
