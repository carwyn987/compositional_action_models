"""On-disk cache of text embeddings: one JSON file of {key: [floats]} (ported from deprecated/)."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np


class DiskEmbeddingCache:
    """A persistent str -> vector cache. Writes are atomic and merge with what is on disk, so
    concurrent runs sharing the file do not lose each other's entries."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.store: dict[str, np.ndarray] = {}
        if self.path.exists():
            try:
                self.store = {key: np.asarray(value, dtype=np.float32) for key, value in json.loads(self.path.read_text()).items()}
            except (json.JSONDecodeError, OSError):
                self.store = {}  # unreadable cache: start fresh rather than fail training

    def get(self, key: str) -> np.ndarray | None:
        return self.store.get(key)

    def set(self, key: str, vector: np.ndarray) -> None:
        self.store[key] = np.asarray(vector, dtype=np.float32)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                for key_on_disk, value in json.loads(self.path.read_text()).items():
                    self.store.setdefault(key_on_disk, np.asarray(value, dtype=np.float32))
            except (json.JSONDecodeError, OSError):
                pass
        fd, partial = tempfile.mkstemp(dir=self.path.parent, suffix=".partial")
        with os.fdopen(fd, "w") as handle:
            json.dump({key: value.tolist() for key, value in self.store.items()}, handle)
        os.replace(partial, self.path)
