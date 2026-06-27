#!/usr/bin/env bash
set -euo pipefail

# Visualize the vanilla (no symbolic embedding) directional push policies.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for SKILL in pushleft pushright pushforward pushbackward; do
  BASENAME="$(python -m symb_model_embeddings.run_naming --skill "${SKILL}" --algo ppo --teacher bc --tag vanilla)"
  MUJOCO_GL=glfw python experimental/rollout_vanilla_skill.py \
    --skill "${SKILL}" \
    --algo ppo \
    --model "models/vanilla/${BASENAME}.zip" \
    --episodes 5
done
