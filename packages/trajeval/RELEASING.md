# Releasing trajeval

A checklist, not automation — this is a pre-1.0, low-frequency-release
project, and a manual checklist is more honest than a version-bump script
that's only ever been run once or twice. Revisit this if releases become
frequent enough that the manual steps start causing mistakes.

## Before every release

1. Everything in `CONTRIBUTING.md`'s checks is green on `main`: `ruff
   format --check`, `ruff check`, `mypy --strict`, `pytest --cov`, `mkdocs
   build --strict`, and (if `apps/web` changed) `lint` / `tsc --noEmit` /
   `build`. In practice this just means: CI is green on `main`.
2. `packages/trajeval/CHANGELOG.md`'s `[Unreleased]` section accurately
   describes everything user-visible since the last release. Move it under
   a new `## [X.Y.Z] - YYYY-MM-DD` heading, leave a fresh empty
   `[Unreleased]` above it.
3. Bump `version` in `packages/trajeval/pyproject.toml` to match. This is
   the *only* place the version is hand-written — `trajeval.__version__`
   and `trajeval --version` both read it back from installed package
   metadata at runtime, so nothing else needs updating.
4. If `apps/web` changed since the last release, rebuild the bundled
   viewer: `python scripts/bundle_webapp.py`. (The release workflow does
   this too, but do it locally first if you want to sanity-check the
   viewer with `trajeval serve` before cutting the release.)
5. Commit: `git commit -am "Release vX.Y.Z"`.

## Cutting the release

1. Tag: `git tag vX.Y.Z && git push origin main --tags`.
2. On GitHub, draft a Release from that tag. Paste the CHANGELOG section
   for this version into the release notes.
3. Publish the release. This is what triggers `.github/workflows/publish.yml`
   — it builds, runs the full test suite as a safety gate, verifies the
   wheel actually contains the bundled viewer and `py.typed`, and publishes
   to PyPI via trusted publishing (OIDC — no token stored in this repo).
4. Confirm on PyPI: <https://pypi.org/project/trajectory-eval/>.

## One-time setup (already done, or needed once before the first release)

- **PyPI trusted publisher**: on pypi.org, the `trajectory-eval` project →
  Publishing → add a trusted publisher for this repo, workflow file
  `publish.yml`, environment `pypi`. For a brand-new project name, PyPI
  supports registering this before the project has ever been published
  ("pending publishers"). Without this, `publish.yml` fails at the publish
  step with an authorization error — it does not silently fall back to a
  token or anything else.
- **GitHub Pages / Vercel**: see `DEPLOYMENT.md` at the repo root — these
  are separate from the PyPI release process above and don't need
  repeating per release.

## If something goes wrong after publishing

PyPI doesn't allow overwriting a published version. If a release is
broken:

1. Fix the bug, bump to the next patch version, release that instead.
2. If the broken version is bad enough to actively steer people away from,
   `pip install trajectory-eval==X.Y.Z` and `yank` it from PyPI's UI —
   yanking hides it from `pip install trajectory-eval` (which resolves to
   the latest non-yanked version) without deleting it, so anyone already
   pinned to that exact version isn't broken.
