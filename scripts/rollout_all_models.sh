#!/usr/bin/env bash
set -euo pipefail

# Run a single trained multi-skill model on every skill it was trained on.
# Prints per-episode results for each skill so you can compare them side by side.
#
# Usage:
#   ./scripts/rollout_all_models.sh models/multi-pickup-putdown_ppo_mock-d32_t-bc_s0.zip
#   SKILLS="pickup putdown" ./scripts/rollout_all_models.sh models/my_model.zip
#   EPISODES=10 ALGO=sac ./scripts/rollout_all_models.sh models/my_model.zip
#
# Arguments:
#   $1           Path to the .zip model file (required)
#
# Environment variables:
#   SKILLS       Ordered skills the model was trained on  default: "pickup putdown"
#   ALGO         sac or ppo                            default: ppo
#   EPISODES     Episodes per skill                    default: 5
#   SEED         Eval seed                             default: 1
#   NO_RENDER    Set to 1 to disable rendering         default: 1

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model.zip>" >&2
  exit 1
fi

MODEL="$1"
SKILLS="${SKILLS:-pickup putdown}"
ALGO="${ALGO:-ppo}"
EPISODES="${EPISODES:-5}"
SEED="${SEED:-1}"
NO_RENDER="${NO_RENDER:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
PYTHON="${REPO_ROOT}/.venv/bin/python"

RENDER_FLAG=""
if [[ "${NO_RENDER}" == "1" ]]; then
  RENDER_FLAG="--no-render"
fi

# shellcheck disable=SC2206
SKILL_LIST=(${SKILLS})

echo "Model: ${MODEL}"
echo "Skills: ${SKILL_LIST[*]}  algo=${ALGO}  episodes=${EPISODES}  seed=${SEED}"
echo

for SKILL in "${SKILL_LIST[@]}"; do
  echo "=== skill: ${SKILL} ==="
  "${PYTHON}" rollout_multiskill.py \
    --model "${MODEL}" \
    --skills "${SKILL_LIST[@]}" \
    --skill "${SKILL}" \
    --algo "${ALGO}" \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    ${RENDER_FLAG}
  echo
done
