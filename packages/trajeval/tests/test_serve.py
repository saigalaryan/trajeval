"""Tests for trajeval.serve.

`_ViewerRequestHandler.translate_path` is the load-bearing logic here and is
testable without ever binding a socket — `http.server` handlers don't
require the network machinery to construct and call individual methods on,
as long as `__init__` (which normally processes a real request) is skipped.
The `serve()` function's actual socket-binding path is covered by a real
end-to-end smoke test instead, not unit tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trajeval.serve import _WEBAPP_DIST, ViewerNotBundledError, _ViewerRequestHandler, serve


def _handler_for(data_dir: Path) -> _ViewerRequestHandler:
    """Construct a handler without running __init__'s request-processing
    side effects (it normally reads from a live socket)."""
    handler = _ViewerRequestHandler.__new__(_ViewerRequestHandler)
    handler.data_dir = data_dir
    return handler


def test_translate_path_root_serves_index_html(tmp_path: Path) -> None:
    handler = _handler_for(tmp_path)
    assert handler.translate_path("/") == str(_WEBAPP_DIST / "index.html")


def test_translate_path_clean_url_maps_to_html_file(tmp_path: Path, monkeypatch) -> None:
    """Next.js static export's clean-URL routing: /run -> run.html — and
    specifically must resolve to the .html file even when a same-named
    *directory* also exists (the App Router's static export emits both a
    run.html file and a run/ directory of RSC payload chunks for the same
    route — this is the exact shape that broke translate_path before it
    checked is_file() instead of exists()).

    Builds its own fixture dist rather than depending on the real bundled
    _webapp_dist, which only exists locally after running
    scripts/bundle_webapp.py — never true on a fresh CI checkout.
    """
    fake_dist = tmp_path / "webapp_dist"
    fake_dist.mkdir()
    (fake_dist / "run.html").write_text("<html></html>", encoding="utf-8")
    (fake_dist / "run").mkdir()
    monkeypatch.setattr("trajeval.serve._WEBAPP_DIST", fake_dist)

    handler = _handler_for(tmp_path)
    result = handler.translate_path("/run")
    assert result == str(fake_dist / "run.html")


def test_translate_path_static_asset_passes_through_unchanged(tmp_path: Path) -> None:
    handler = _handler_for(tmp_path)
    assert handler.translate_path("/favicon.ico") == str(_WEBAPP_DIST / "favicon.ico")


def test_translate_path_data_prefix_routes_to_data_dir(tmp_path: Path) -> None:
    handler = _handler_for(tmp_path)
    assert handler.translate_path("/data/latest.json") == str((tmp_path / "latest.json").resolve())


def test_translate_path_data_prefix_supports_nested_paths(tmp_path: Path) -> None:
    handler = _handler_for(tmp_path)
    assert handler.translate_path("/data/sub/dir/file.json") == str(
        (tmp_path / "sub" / "dir" / "file.json").resolve()
    )


def test_translate_path_data_prefix_url_decodes(tmp_path: Path) -> None:
    handler = _handler_for(tmp_path)
    assert handler.translate_path("/data/my%20run.json") == str(
        (tmp_path / "my run.json").resolve()
    )


def test_translate_path_query_string_ignored(tmp_path: Path) -> None:
    handler = _handler_for(tmp_path)
    assert handler.translate_path("/data/latest.json?foo=bar") == str(
        (tmp_path / "latest.json").resolve()
    )


def test_translate_path_degenerate_unknown_clean_url_falls_through(tmp_path: Path) -> None:
    """A path with no matching file and no .html sibling just returns the
    literal (nonexistent) candidate — the base handler's 404 machinery
    takes it from there; translate_path itself never raises."""
    handler = _handler_for(tmp_path)
    result = handler.translate_path("/this-route-does-not-exist")
    assert result == str(_WEBAPP_DIST / "this-route-does-not-exist")


# ---------------------------------------------------------------------------
# serve() — argument validation (no socket actually bound in these)
# ---------------------------------------------------------------------------


def test_serve_degenerate_missing_results_dir_raises(tmp_path: Path, monkeypatch) -> None:
    # Needs a bundled-viewer fixture so serve() gets past its "is the
    # viewer bundled" check and actually reaches the results_dir check
    # this test is about — see test_serve_raises_viewer_not_bundled_when_dist_missing
    # for the inverse case.
    fake_dist = tmp_path / "webapp_dist"
    fake_dist.mkdir()
    monkeypatch.setattr("trajeval.serve._WEBAPP_DIST", fake_dist)

    with pytest.raises(FileNotFoundError):
        serve(tmp_path / "does-not-exist", port=0, open_browser=False)


def test_serve_raises_viewer_not_bundled_when_dist_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("trajeval.serve._WEBAPP_DIST", tmp_path / "nonexistent-dist")
    with pytest.raises(ViewerNotBundledError, match="bundle_webapp.py"):
        serve(tmp_path, port=0, open_browser=False)
