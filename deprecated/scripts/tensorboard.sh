#!/usr/bin/env bash
set -euo pipefail

# Launch TensorBoard on the training logs.
#
# Both training entrypoints pass `tensorboard_log=<logdir>` to Stable-Baselines3,
# so event files land under the run log directory (default: runs/). Point this at
# the same directory you trained with.
#
# Usage:
#   ./scripts/tensorboard.sh
#   LOGDIR=runs PORT=6007 ./scripts/tensorboard.sh
#   ./scripts/tensorboard.sh --samples_per_plugin 'scalars=2000'   # extra flags
#
# Environment variables:
#   LOGDIR   directory of TB event files   default: runs
#   PORT     port to serve on              default: 6006
#   HOST     bind address                  default: localhost

LOGDIR="${LOGDIR:-runs}"
PORT="${PORT:-6006}"
HOST="${HOST:-localhost}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v tensorboard >/dev/null 2>&1; then
  echo "ERROR: 'tensorboard' not found. Install it with: pip install tensorboard" >&2
  exit 1
fi

echo "Serving TensorBoard for '${LOGDIR}' at http://${HOST}:${PORT}"
exec tensorboard --logdir "${LOGDIR}" --port "${PORT}" --host "${HOST}" "$@"
