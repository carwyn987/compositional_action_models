"""Symbolic action-model embedding subsystem.

Public surface used by training/rollout entrypoints. Everything here is
independent of ``fetch_blockworld`` so the embedding scheme can evolve (toward
learnable component embedders + aggregators) without touching the env package.
"""

from .action_model import EMBED_SOURCES, SymbolicActionModel
from .cache import DiskEmbeddingCache
from .embedders import (
    EmbedderConfig,
    MockEmbedder,
    SymbolicEmbedder,
    TextEmbedder,
    add_embedder_cli_args,
    build_embedder,
    embedder_config_from_args,
)
from .naming import basename_from_args, embed_tag, model_basename
from .obs_symb_wrapper import EnvWrapper

__all__ = [
    "EMBED_SOURCES",
    "SymbolicActionModel",
    "DiskEmbeddingCache",
    "EmbedderConfig",
    "MockEmbedder",
    "SymbolicEmbedder",
    "TextEmbedder",
    "EnvWrapper",
    "add_embedder_cli_args",
    "build_embedder",
    "embedder_config_from_args",
    "basename_from_args",
    "embed_tag",
    "model_basename",
]
