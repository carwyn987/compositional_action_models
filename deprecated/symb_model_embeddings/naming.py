"""Single source of truth for run/model file names.

A trained policy's file name encodes the major option choices that change the
resulting artifact, so runs with different settings never overwrite each other:

* ``skill`` and ``algo``
* the symbolic embedding (kind / text source / backend / dimensionality), which
  also determines the observation dimensionality
* the teacher (behaviour-cloning warm-start vs. none)
* the random ``seed``

Both the Python training entrypoints and the bash sweep drivers derive names
from here -- the drivers call ``python -m symb_model_embeddings.run_naming``
(a thin CLI over these functions) so the two can never drift apart.
"""

from __future__ import annotations

import argparse

from .embedders import EmbedderConfig, embedder_config_from_args


def embed_tag(config: EmbedderConfig) -> str:
    """Short, filename-safe tag describing an embedding configuration.

    A ``-train`` suffix marks a trainable (learned) embedding, so trainable and
    frozen runs of the same scheme save to distinct files.
    """
    if config.kind == "mock":
        tag = f"mock-d{config.size}"
    else:
        tag = f"text-{config.source}-{config.backend}-d{config.size}"
    if config.trainable:
        tag += "-train"
    return tag


def model_basename(
    skill: str,
    algo: str,
    tag: str,
    teacher: str,
    seed: int,
) -> str:
    """Build the (extension-less) base name shared by a run's model and logs."""
    return f"{skill}_{algo}_{tag}_t-{teacher}_s{seed}"


def basename_from_args(args: argparse.Namespace) -> str:
    """Derive the run base name from parsed CLI args.

    Uses ``args.tag`` when provided (e.g. the vanilla, no-embedding entrypoints
    pass ``--tag vanilla``); otherwise derives the embedding tag from the
    embedder options.
    """
    tag = getattr(args, "tag", None) or embed_tag(embedder_config_from_args(args))
    return model_basename(args.skill, args.algo, tag, args.teacher, args.seed)
