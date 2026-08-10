"""trajeval serve: run the bundled web viewer locally, pointed at a results
directory — zero Node.js dependency for the end user. The viewer
(`apps/web`) is built once at release time (see `scripts/bundle_webapp.py`)
and shipped inside the `trajeval` wheel as static files; this module is
just Python's own `http.server` serving two directory trees from one
process: the bundled viewer at `/`, and the results directory at `/data/*`.

No database, no hosted backend, no auth — same design rule as everywhere
else in this project. This is a *local* convenience server, not a service
meant to be exposed beyond localhost.
"""

from __future__ import annotations

import functools
import http.server
import socketserver
import urllib.parse
import webbrowser
from pathlib import Path

_WEBAPP_DIST = Path(__file__).parent / "_webapp_dist"

# Extensions that should be served as-is (never have ".html" appended for
# Next.js's clean-URL routing).
_STATIC_EXTENSIONS = (
    ".html",
    ".js",
    ".css",
    ".json",
    ".ico",
    ".svg",
    ".txt",
    ".map",
    ".woff",
    ".woff2",
)


class ViewerNotBundledError(RuntimeError):
    """Raised when this install has no bundled web viewer to serve."""


class _ViewerRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the bundled viewer at `/` and `data_dir` at `/data/*`.

    Overrides `translate_path` (rather than passing `directory=` to the
    base class) because we're dispatching between two separate directory
    trees on one server, which the base class's single-`directory` model
    doesn't support.
    """

    def __init__(self, *args: object, data_dir: Path, **kwargs: object) -> None:
        self.data_dir = data_dir
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def translate_path(self, path: str) -> str:
        parsed = urllib.parse.unquote(urllib.parse.urlparse(path).path)

        if parsed.startswith("/data/"):
            rel = parsed[len("/data/") :].lstrip("/")
            return str((self.data_dir / rel).resolve())

        rel = parsed.lstrip("/") or "index.html"
        candidate = _WEBAPP_DIST / rel
        if not candidate.is_file() and not rel.endswith(_STATIC_EXTENSIONS):
            # Next.js static export's clean-URL routing: /run -> run.html.
            # Note candidate.is_file() (not .exists()): the App Router's
            # static export also emits a same-named *directory* per route
            # (RSC payload chunks like __next._full.txt) alongside run.html,
            # so a plain existence check would match that directory instead
            # of ever falling through to the actual HTML page.
            html_candidate = _WEBAPP_DIST / f"{rel}.html"
            if html_candidate.exists():
                candidate = html_candidate
        return str(candidate)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - matches base signature
        pass  # quiet by default — serve() prints its own one-line banner instead


def serve(
    results_dir: str | Path,
    *,
    file: str | None = None,
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    """Serve the bundled web viewer at ``http://localhost:<port>``, pointed
    at `results_dir`.

    If `file` is given (a filename inside `results_dir`), opens directly to
    that RunResult via `?src=/data/<file>`; otherwise opens the viewer's
    loader page and you pick a file yourself (drag-and-drop still works —
    it reads the file straight off disk in the browser, independent of the
    `/data/` route).

    Raises `ViewerNotBundledError` if this install has no bundled viewer —
    the common case being a source checkout where
    `python scripts/bundle_webapp.py` hasn't been run. Blocks until
    Ctrl+C.
    """
    if not _WEBAPP_DIST.exists():
        raise ViewerNotBundledError(
            "The bundled web viewer isn't present in this install. If you're running from "
            "source, build it first: `python scripts/bundle_webapp.py` (requires Node.js). "
            "A published trajeval wheel bundles it automatically."
        )

    data_dir = Path(results_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"results directory not found: {data_dir}")

    handler = functools.partial(_ViewerRequestHandler, data_dir=data_dir)

    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://localhost:{port}/"
        if file:
            url += f"?src=/data/{urllib.parse.quote(file)}"

        print(f"Serving trajeval viewer at {url}")
        print(f"  results directory: {data_dir}")
        print("  Ctrl+C to stop")

        if open_browser:
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
