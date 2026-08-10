"""Content-addressed cache for judge responses.

Judged metrics are re-run often — once during development, again in CI, again
when a new metric is added to the same run. Without caching, every re-run
re-pays for identical judge calls on unchanged trajectories. The cache key is
a hash of exactly what was sent (prompt, system, model), so any change to the
inputs is a cache miss, and any re-run of unchanged inputs is a hit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def cache_key(*, prompt: str, system: str | None, model: str) -> str:
    payload = json.dumps({"prompt": prompt, "system": system, "model": model}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JudgeCache:
    """A dict-backed cache, optionally persisted to a JSON file on disk.

    Persistence is simple and re-writes the whole file on every ``set`` —
    fine at the call volumes a harness run produces, and it means a run
    that's interrupted mid-way doesn't lose what it already cached.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._store: dict[str, str] = {}
        if self._path is not None and self._path.exists():
            self._store = json.loads(self._path.read_text(encoding="utf-8"))

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        self._store[key] = value
        if self._path is not None:
            self._path.write_text(json.dumps(self._store, indent=2), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._store)
