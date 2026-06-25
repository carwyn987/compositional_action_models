"""Pretrained text-embedding backends.

A backend turns a string into a fixed pretrained vector. Results are routed
through :class:`DiskEmbeddingCache` so each unique string is embedded once.

Two backends ship today:

* :class:`OpenAIBackend` -- real pretrained embeddings via the OpenAI API.
  Requires the ``openai`` package and an ``OPENAI_API_KEY``.
* :class:`HashBackend` -- deterministic, dependency-free offline embeddings.
  Lets the full pipeline run (and ablations stay reproducible) with no network
  or API key. Not semantically meaningful; intended as a stand-in/baseline.

New backends only need to implement :class:`TextEmbeddingBackend`.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np

from .cache import DiskEmbeddingCache, get_default_cache

TEXT_BACKENDS = ("openai", "hash")
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


@runtime_checkable
class TextEmbeddingBackend(Protocol):
    """Maps a string to a 1-D pretrained embedding vector."""

    #: Stable identifier used as part of the cache key.
    id: str
    #: Output dimensionality.
    dim: int

    def embed_text(self, text: str) -> np.ndarray: ...


def _cache_key(backend_id: str, dim: int, text: str) -> str:
    return f"{backend_id}|dim={dim}|{text}"


class HashBackend:
    """Deterministic offline embedding from a hash of the text.

    Produces a reproducible unit-norm vector. Useful as an offline default and
    as a control baseline (no semantic content) for ablations.
    """

    def __init__(self, dim: int = 32, cache: DiskEmbeddingCache | None = None) -> None:
        self.dim = int(dim)
        self.id = f"hash-{self.dim}"
        self._cache = cache if cache is not None else get_default_cache()

    def embed_text(self, text: str) -> np.ndarray:
        key = _cache_key(self.id, self.dim, text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Seed an RNG deterministically from the text so the vector is stable
        # across processes and machines.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self.dim).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm

        self._cache.set(key, vector)
        return vector


class OpenAIBackend:
    """Pretrained sentence embeddings via the OpenAI embeddings API.

    ``dim`` is passed through as the ``dimensions`` request parameter (supported
    by ``text-embedding-3-*``), so the same configured size works across
    backends. Set ``dim=0`` to use the model's native dimensionality.
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        dim: int = 32,
        cache: DiskEmbeddingCache | None = None,
    ) -> None:
        self.model = model
        self.dim = int(dim)
        self.id = f"openai-{model}-{self.dim}"
        self._cache = cache if cache is not None else get_default_cache()
        self._client = None  # Lazily constructed on first real call.

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on env
                raise RuntimeError(
                    "The 'openai' package is required for the OpenAI embedding "
                    "backend. Install it with `pip install openai`, or use "
                    "`--embed-backend hash` for an offline run."
                ) from exc
            self._client = OpenAI()
        return self._client

    def embed_text(self, text: str) -> np.ndarray:
        key = _cache_key(self.id, self.dim, text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        kwargs = {"model": self.model, "input": text}
        if self.dim > 0:
            kwargs["dimensions"] = self.dim
        response = self._get_client().embeddings.create(**kwargs)
        vector = np.asarray(response.data[0].embedding, dtype=np.float32)

        self._cache.set(key, vector)
        return vector


def build_text_backend(
    backend: str,
    *,
    dim: int,
    model: str = DEFAULT_OPENAI_MODEL,
    cache: DiskEmbeddingCache | None = None,
) -> TextEmbeddingBackend:
    if backend == "openai":
        return OpenAIBackend(model=model, dim=dim, cache=cache)
    if backend == "hash":
        return HashBackend(dim=dim, cache=cache)
    raise ValueError(
        f"Unknown text backend {backend!r}. Choices: {list(TEXT_BACKENDS)}"
    )
