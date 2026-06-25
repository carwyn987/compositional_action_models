#!/usr/bin/env bash
set -euo pipefail

# Visualize the vanilla (no symbolic embedding) directional push policies.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for SKILL in pushleft pushright pushforward pushbackward; do
  MUJOCO_GL=glfw python experimental/rollout_vanilla_skill.py \
    --skill "${SKILL}" \
    --algo ppo \
    --model "models/vanilla/${SKILL}_ppo.zip" \
    --episodes 5
done
