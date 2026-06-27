"""On-disk key/value cache for pretrained text embeddings.

A single JSON file holds ``{key: [floats...]}`` so that expensive pretrained
lookups (e.g. OpenAI embedding calls) are made at most once per unique key.
Keys are opaque strings chosen by the caller (typically backend id + text).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

# Default cache file lives next to this package so it persists across runs.
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "embedding_cache.json"


class DiskEmbeddingCache:
    """A persistent ``str -> np.ndarray`` cache backed by a JSON file."""

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_CACHE_PATH
        self._store: dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            # Corrupt/unreadable cache: start fresh rather than crash training.
            self._store = {}
            return
        self._store = {
            key: np.asarray(value, dtype=np.float32) for key, value in raw.items()
        }

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def get(self, key: str) -> np.ndarray | None:
        return self._store.get(key)

    def set(self, key: str, vector: np.ndarray) -> None:
        self._store[key] = np.asarray(vector, dtype=np.float32)
        self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Merge with whatever is currently on disk so concurrent writers (or
        # other live cache instances) don't clobber each other's keys. Our own
        # in-memory entries win on conflict.
        if self.path.exists():
            try:
                on_disk = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                on_disk = {}
            for key, value in on_disk.items():
                self._store.setdefault(key, np.asarray(value, dtype=np.float32))
        serializable = {key: value.tolist() for key, value in self._store.items()}
        # Atomic write so a crash mid-flush cannot corrupt the cache file.
        fd, tmp_name = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(serializable, handle)
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)


_DEFAULT_CACHE: DiskEmbeddingCache | None = None


def get_default_cache() -> DiskEmbeddingCache:
    """Return the process-global cache backed by ``DEFAULT_CACHE_PATH``.

    Backends share this single instance so all embeddings made within a run go
    through one in-memory dict (and one file), matching the intended global
    key/value semantics.
    """
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = DiskEmbeddingCache()
    return _DEFAULT_CACHE
