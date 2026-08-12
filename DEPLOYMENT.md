# Deployment

One-time hosting setup for this repo's two public-facing pieces: the docs
site and the web viewer. Neither is required to *use* trajeval — both are
optional, convenience-only surfaces (the library works standalone, and the
web viewer runs perfectly well locally via `trajeval serve`). See
`packages/trajeval/RELEASING.md` for the separate, per-release PyPI publish
checklist.

## Docs (GitHub Pages) — automated

`.github/workflows/docs.yml` builds `packages/trajeval/docs/` with MkDocs
and pushes it to a `gh-pages` branch on every push to `main` that touches
docs-relevant paths. One manual step required once the repo exists on
GitHub, since GitHub Pages can't be configured from a workflow file:

1. Push to `main` at least once so the `docs.yml` workflow runs and creates
   the `gh-pages` branch.
2. On GitHub: **Settings → Pages → Build and deployment → Source**: "Deploy
   from a branch". **Branch**: `gh-pages` / `/ (root)`. Save.
3. The site is live at `https://saigalaryan03.github.io/trajeval/` within a
   few minutes (matches `site_url` in `packages/trajeval/mkdocs.yml`).

## Web viewer (Vercel) — manual, one-time

`apps/web` is a static Next.js export (`output: "export"` in
`next.config.ts`) — Vercel serves this natively, no framework config
changes needed. What *does* need manual setup, because this is a monorepo
and it requires either dashboard access or a Vercel API token this session
doesn't have:

1. On [vercel.com](https://vercel.com), **Add New → Project**, import the
   `saigalaryan03/trajeval` GitHub repo.
2. In the import screen's **Root Directory** setting, set it to `apps/web`
   — this is the one setting that matters for a monorepo; without it,
   Vercel looks for `package.json` at the repo root and won't find the
   Next.js app.
3. Framework preset should auto-detect as **Next.js**. Build command,
   output directory, and install command can stay at their Next.js
   defaults — no `vercel.json` is needed.
4. Deploy. Every push to `main` (or a PR, for a preview URL) redeploys
   automatically once the project is connected — no additional workflow
   file required, Vercel's own GitHub integration handles it.

**No environment variables are needed.** The viewer is entirely
client-side — it reads a `RunResult` JSON the user loads themselves
(drag-and-drop, file picker, or a `?src=` query param); there's no backend,
API key, or database for this app to talk to. That's a deliberate design
rule, not something this deployment happened to not need yet.

### Why not automate this in a workflow file?

Connecting a GitHub repo to a new Vercel project requires either interactive
OAuth in Vercel's dashboard or a `VERCEL_TOKEN`/`VERCEL_ORG_ID`/
`VERCEL_PROJECT_ID` secret set from a project that already exists — both are
account-holder actions. Once the project exists (step 1–4 above, done once),
Vercel's own GitHub App handles every subsequent deploy with zero workflow
file on this repo's side.
