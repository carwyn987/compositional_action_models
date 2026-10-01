#!/usr/bin/env bash
set -euo pipefail

# Launch TensorBoard on the training runs.
#
# main.py writes each run's TensorBoard logs to <output-dir>/<run name>/tensorboard/
# (default output dir: outputs/), so pointing TensorBoard at outputs/ shows every run.
#
# Usage:
#   ./scripts/start_tensorboard.sh
#   LOGDIR=outputs/pickup-putdown_sac_multi-hot_shaped_seed0 PORT=6007 ./scripts/start_tensorboard.sh
#   ./scripts/start_tensorboard.sh --samples_per_plugin 'scalars=2000'   # extra TensorBoard flags
#
# Environment variables:
#   LOGDIR   directory of runs or event files   default: outputs
#   PORT     port to serve on                   default: 6006
#   HOST     bind address                       default: localhost

LOGDIR="${LOGDIR:-outputs}"
PORT="${PORT:-6006}"
HOST="${HOST:-localhost}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Prefer the project's virtual environment, so activating it is not required.
if [[ -x .venv/bin/tensorboard ]]; then
  TENSORBOARD=.venv/bin/tensorboard
elif command -v tensorboard >/dev/null 2>&1; then
  TENSORBOARD=tensorboard
else
  echo "ERROR: 'tensorboard' not found. Install the requirements: pip install -r requirements.txt" >&2
  exit 1
fi

if [[ ! -d "${LOGDIR}" ]]; then
  echo "ERROR: log directory '${LOGDIR}' does not exist; train first (./main.py ...) or set LOGDIR" >&2
  exit 1
fi

echo "Serving TensorBoard for '${LOGDIR}' at http://${HOST}:${PORT}"
exec "${TENSORBOARD}" --logdir "${LOGDIR}" --port "${PORT}" --host "${HOST}" "$@"
