"""Symbolic-action-model embedders and a small config/CLI surface.

An embedder maps a :class:`SymbolicActionModel` to a fixed 1-D ``np.ndarray``
that downstream code injects into the observation (see ``obs_symb_wrapper``).

Today there are two:

* :class:`MockEmbedder` -- the original random-per-skill vector (default; keeps
  existing behaviour unchanged).
* :class:`TextEmbedder` -- a non-trainable pretrained word/sentence embedding of
  either the operator *name* alone or the *full action-model string*.

This module is the single integration point for entrypoints: build a config
from argparse via :func:`add_embedder_cli_args` / :func:`embedder_config_from_args`,
then :func:`build_embedder`. The abstraction (``SymbolicEmbedder.embed`` taking a
``SymbolicActionModel``) is deliberately shaped to host the planned
component-wise embedders + aggregators without changing callers.
"""

from __future__ import annotations

import argparse
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .action_model import EMBED_SOURCES, SymbolicActionModel
from .backends import (
    DEFAULT_OPENAI_MODEL,
    TEXT_BACKENDS,
    build_text_backend,
)
from .cache import DiskEmbeddingCache

EMBEDDER_KINDS = ("mock", "text")
DEFAULT_EMBED_SIZE = 32


@dataclass(frozen=True)
class EmbedderConfig:
    """Fully describes how to build an embedder (ablation-friendly)."""

    kind: str = "mock"            # mock | text
    source: str = "name"          # name | action_model  (text only)
    backend: str = "openai"       # openai | hash         (text only)
    model: str = DEFAULT_OPENAI_MODEL
    size: int = DEFAULT_EMBED_SIZE
    trainable: bool = False

    def validate(self) -> None:
        if self.kind not in EMBEDDER_KINDS:
            raise ValueError(
                f"Unknown embedder kind {self.kind!r}. Choices: {list(EMBEDDER_KINDS)}"
            )
        if self.source not in EMBED_SOURCES:
            raise ValueError(
                f"Unknown embed source {self.source!r}. Choices: {list(EMBED_SOURCES)}"
            )
        if self.backend not in TEXT_BACKENDS:
            raise ValueError(
                f"Unknown embed backend {self.backend!r}. Choices: {list(TEXT_BACKENDS)}"
            )


class SymbolicEmbedder(ABC):
    """Maps a symbolic action model to a fixed-length vector."""

    #: Whether the produced embedding is intended to be learned downstream.
    trainable: bool = False

    @abstractmethod
    def embed(self, action_model: SymbolicActionModel) -> np.ndarray:
        """Return a 1-D ``np.ndarray`` embedding for ``action_model``."""


class MockEmbedder(SymbolicEmbedder):
    """A random vector that is fixed *per operator name*.

    Carries no semantic content (a control / baseline), but is deterministic and
    distinct for each skill, so it can still serve as the skill identifier when a
    single policy is trained on several skills at once (multi-skill training).
    """

    def __init__(self, size: int = DEFAULT_EMBED_SIZE) -> None:
        self.size = int(size)
        self.trainable = False

    def embed(self, action_model: SymbolicActionModel) -> np.ndarray:
        # Seed an RNG from the operator name so the vector is stable across
        # processes/machines yet differs between skills.
        digest = hashlib.sha256(action_model.name.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.size).astype(np.float32)


class TextEmbedder(SymbolicEmbedder):
    """Non-trainable pretrained embedding of the action's name or full string."""

    def __init__(self, backend, source: str = "name") -> None:
        if source not in EMBED_SOURCES:
            raise ValueError(
                f"Unknown embed source {source!r}. Choices: {list(EMBED_SOURCES)}"
            )
        self.backend = backend
        self.source = source
        self.trainable = False

    def embed(self, action_model: SymbolicActionModel) -> np.ndarray:
        text = action_model.text(self.source)
        return np.asarray(self.backend.embed_text(text), dtype=np.float32)


def build_embedder(
    config: EmbedderConfig,
    *,
    cache: DiskEmbeddingCache | None = None,
) -> SymbolicEmbedder:
    """Construct an embedder from a config.

    Trainable embeddings are part of the planned learnable scheme and are not
    wired into the policy yet; requesting one fails loudly rather than silently
    producing a frozen vector.
    """
    config.validate()
    if config.trainable:
        raise NotImplementedError(
            "Trainable symbolic embeddings are not implemented yet. They require "
            "a custom policy feature extractor (the planned learnable-embedding "
            "scheme); use --embed-trainable false for now."
        )

    if config.kind == "mock":
        return MockEmbedder(size=config.size)

    backend = build_text_backend(
        config.backend, dim=config.size, model=config.model, cache=cache
    )
    return TextEmbedder(backend, source=config.source)


# --------------------------------------------------------------------------- #
# Shared argparse surface so entrypoints stay tiny.
# --------------------------------------------------------------------------- #
def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes", "y"):
        return True
    if value.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def add_embedder_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add the symbolic-embedding options to an argument parser."""
    group = parser.add_argument_group("symbolic embedding")
    group.add_argument(
        "--embedder",
        choices=list(EMBEDDER_KINDS),
        default="mock",
        help="Embedding scheme: 'mock' random vector or 'text' pretrained.",
    )
    group.add_argument(
        "--embed-source",
        choices=list(EMBED_SOURCES),
        default="name",
        help="Text source for the 'text' embedder: operator name or full action model.",
    )
    group.add_argument(
        "--embed-backend",
        choices=list(TEXT_BACKENDS),
        default="openai",
        help="Pretrained text backend ('openai' real, 'hash' offline baseline).",
    )
    group.add_argument(
        "--embed-model",
        default=DEFAULT_OPENAI_MODEL,
        help="Model id for the OpenAI text backend.",
    )
    group.add_argument(
        "--embed-size",
        type=int,
        default=DEFAULT_EMBED_SIZE,
        help="Embedding dimensionality (OpenAI uses it as 'dimensions').",
    )
    group.add_argument(
        "--embed-trainable",
        type=_str2bool,
        default=False,
        help="Whether the embedding is trainable (not yet implemented).",
    )


def embedder_config_from_args(args: argparse.Namespace) -> EmbedderConfig:
    return EmbedderConfig(
        kind=args.embedder,
        source=args.embed_source,
        backend=args.embed_backend,
        model=args.embed_model,
        size=args.embed_size,
        trainable=args.embed_trainable,
    )
