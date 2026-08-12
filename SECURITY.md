# Security Policy

trajeval has no hosted service and holds no user data beyond whatever you
run it against locally — the library, CLI, and web viewer are all
local-only by design (see the README's "no database, no hosted backend"
rule). That significantly narrows what a "security issue" here can even
be, but a few things are still worth reporting responsibly rather than as
a public issue:

- A vulnerability in how `trajeval.judge` handles or logs judge responses
  (e.g. something in the judge cache that shouldn't be written to disk in
  plaintext, beyond what's already documented in `trajeval.config`'s
  `judge_cache_path` field).
- A vulnerability in `trajeval serve`'s local HTTP server (e.g. a path
  traversal past `_ViewerRequestHandler.translate_path`'s intended
  boundaries) — it's meant to be `127.0.0.1`-only, so a break of that
  assumption is a real issue.
- A supply-chain concern in a dependency this project pulls in.

## Reporting

Please **do not** open a public GitHub issue for a security report. Instead,
email the maintainer directly (see `pyproject.toml`'s `authors` field for
the current contact) with:

- A description of the issue and its impact.
- Steps to reproduce, or a minimal example.
- Anything you'd like credited once it's fixed.

You should get an acknowledgment within a few days. This is a solo/small
open-source project without a formal SLA, but security reports get
priority over everything else in the backlog.

## Scope

Out of scope: issues that require an attacker to already have local code
execution on the machine running `trajeval` (at that point they don't need
a vulnerability in this project), and issues in `apps/web`'s hosted
deployment (Vercel), which is a convenience mirror of the same static,
client-side code that ships with the package — report those the same way,
but they carry the same "no backend, no server-side state" scope limits.
