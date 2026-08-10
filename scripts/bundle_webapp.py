#!/usr/bin/env python3
"""Build apps/web's static export and copy it into
packages/trajeval/src/trajeval/_webapp_dist/, where it's bundled as package
data so `trajeval serve` works with zero Node dependency for end users.

Run this from the repo root before cutting a release, and any time apps/web
changes and you want `trajeval serve` to reflect it locally:

    python scripts/bundle_webapp.py

This is intentionally a manual step, not a build hook wired into
`packages/trajeval`'s own build — building the Next.js app requires Node,
and `pip install trajeval` must never require Node to succeed. Automating
this into CI (build once, commit the dist, or build in the release
workflow) is a reasonable follow-up; this script is what that automation
would call.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_APP_DIR = REPO_ROOT / "apps" / "web"
BUILT_EXPORT_DIR = WEB_APP_DIR / "out"
BUNDLED_DEST_DIR = REPO_ROOT / "packages" / "trajeval" / "src" / "trajeval" / "_webapp_dist"


def main() -> int:
    if not WEB_APP_DIR.exists():
        print(f"error: {WEB_APP_DIR} not found — run this from the repo root", file=sys.stderr)
        return 1

    print(f"Building {WEB_APP_DIR} (npm run build)...")
    result = subprocess.run(["npm", "run", "build"], cwd=WEB_APP_DIR, shell=(sys.platform == "win32"))
    if result.returncode != 0:
        print("error: npm run build failed", file=sys.stderr)
        return result.returncode

    if not BUILT_EXPORT_DIR.exists():
        print(f"error: expected static export at {BUILT_EXPORT_DIR} but it doesn't exist", file=sys.stderr)
        return 1

    if BUNDLED_DEST_DIR.exists():
        shutil.rmtree(BUNDLED_DEST_DIR)
    shutil.copytree(BUILT_EXPORT_DIR, BUNDLED_DEST_DIR)

    size_mb = sum(f.stat().st_size for f in BUNDLED_DEST_DIR.rglob("*") if f.is_file()) / 1_000_000
    print(f"Bundled {BUNDLED_DEST_DIR} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
