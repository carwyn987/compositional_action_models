"""CLI front-end for :mod:`symb_model_embeddings.naming`.

Prints the base name for a training run (shared by its model file and logs) so
the bash sweep drivers and the Python entrypoints agree on names. The naming
logic itself lives in :mod:`naming`; this module is intentionally a thin wrapper
so it can be invoked as ``python -m symb_model_embeddings.run_naming ...``.
"""

from __future__ import annotations

import argparse

from .embedders import add_embedder_cli_args
from .naming import basename_from_args


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the base name for a training run (model + logs)."
    )
    parser.add_argument("--skill", required=True)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--teacher", default="bc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tag",
        default=None,
        help="Override the embedding tag (e.g. 'vanilla' for no-embedding runs).",
    )
    add_embedder_cli_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    print(basename_from_args(_parse_args()))
