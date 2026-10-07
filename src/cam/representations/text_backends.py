"""Text embedding backends: a string -> a fixed-length vector.

    MockTextBackend    deterministic, offline stand-in: signed feature hashing of character n-grams,
                       so texts sharing substrings get similar vectors (no semantics beyond that)
    OpenAIBackend      pretrained embeddings from the OpenAI API (needs the `openai` package); the API key
                       comes from OPENAI_API_KEY, else from the file named by OPENAI_API_KEY_FILE
                       (default ~/.secrets/openai_api_key, containing only the key); `dimensions` sets
                       the output size for text-embedding-3-* models
    CachedTextBackend  wraps a backend with a DiskEmbeddingCache, so each unique text is embedded once
                       (and API results stay fixed across runs)

New backends implement TextEmbeddingBackend (id, dim, embed_text).
"""

import hashlib
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from cam.representations.embedding_cache import DiskEmbeddingCache


class TextEmbeddingBackend(ABC):
    id: str  # identifies the backend and its settings; part of the cache key
    dim: int

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """float32 vector of length dim."""


class MockTextBackend(TextEmbeddingBackend):
    """Signed feature hashing of the text's character n-grams into dim buckets, L2-normalised.

    Each n-gram adds +1 or -1 to one bucket, both chosen by SHA-256 of the n-gram, so the vector is
    the same in every process. Texts sharing many n-grams have high cosine similarity, roughly like
    a text embedding of surface form; it carries no meaning beyond shared substrings.
    """

    def __init__(self, dim: int, ngram: int = 3):
        self.dim, self.ngram = dim, ngram
        self.id = f"mock-char{ngram}gram-{dim}"

    def embed_text(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float64)
        for i in range(max(len(text) - self.ngram + 1, 1)):
            digest = hashlib.sha256(text[i : i + self.ngram].encode()).digest()
            bucket = int.from_bytes(digest[:8], "little") % self.dim
            vector[bucket] += 1.0 if digest[8] & 1 else -1.0
        norm = np.linalg.norm(vector)
        return (vector / norm if norm > 0 else vector).astype(np.float32)


DEFAULT_OPENAI_API_KEY_FILE = "~/.secrets/openai_api_key"


def openai_api_key() -> str:
    """OPENAI_API_KEY if set, else the contents (stripped) of OPENAI_API_KEY_FILE or the default file.
    The key is never logged or included in error messages."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    path = Path(os.environ.get("OPENAI_API_KEY_FILE", DEFAULT_OPENAI_API_KEY_FILE)).expanduser()
    if not path.is_file():
        raise RuntimeError(f"no OpenAI API key: set OPENAI_API_KEY or put the key in {path}")
    return path.read_text().strip()


class OpenAIBackend(TextEmbeddingBackend):
    """Embeddings from the OpenAI API, e.g. text-embedding-3-small, truncated to dim via `dimensions`."""

    def __init__(self, dim: int, model: str = "text-embedding-3-small"):
        self.dim, self.model = dim, model
        self.id = f"openai-{model}-{dim}"
        self.client = None

    def embed_text(self, text: str) -> np.ndarray:
        if self.client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError("the openai backend needs `pip install openai`") from error
            self.client = OpenAI(api_key=openai_api_key())
        response = self.client.embeddings.create(model=self.model, input=text, dimensions=self.dim)
        return np.asarray(response.data[0].embedding, dtype=np.float32)


class CachedTextBackend(TextEmbeddingBackend):
    def __init__(self, backend: TextEmbeddingBackend, cache: DiskEmbeddingCache):
        self.backend, self.cache = backend, cache
        self.id, self.dim = backend.id, backend.dim

    def embed_text(self, text: str) -> np.ndarray:
        key = f"{self.id}|{text}"
        vector = self.cache.get(key)
        if vector is None:
            vector = self.backend.embed_text(text)
            self.cache.set(key, vector)
        return vector


TEXT_BACKENDS = {"mock": MockTextBackend, "openai": OpenAIBackend}
