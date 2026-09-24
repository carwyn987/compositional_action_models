#!/usr/bin/env bash
set -euo pipefail

# Train + evaluate every skill using the MOCK embedding (a fixed random
# per-skill vector, independent of the symbolic action model). This is the
# baseline that ignores symbolic content.
#
# All variables understood by per_skill/train_all_and_eval.sh are forwarded,
# e.g.:
#   TIMESTEPS=200000 ALGO=ppo ./scripts/embeddings/mock/train_all.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env EMBEDDER=mock "${HERE}/../../per_skill/train_all_and_eval.sh" "$@"
