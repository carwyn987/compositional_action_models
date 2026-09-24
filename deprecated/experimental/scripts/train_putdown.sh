#!/usr/bin/env bash
set -euo pipefail

# Vanilla (no symbolic embedding) putdown training + evaluation.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python experimental/train_vanilla_skill.py \
  --skill putdown \
  --algo ppo \
  --teacher bc \
  --teacher-rollouts 500 \
  --teacher-gradient-steps 20000 \
  --teacher-batch-size 1024 \
  --teacher-lr 1e-4 \
  --timesteps 100000 \
  --models-dir models/vanilla

BASENAME="$(python -m symb_model_embeddings.run_naming --skill putdown --algo ppo --teacher bc --tag vanilla)"
MUJOCO_GL=glfw python experimental/rollout_vanilla_skill.py \
  --skill putdown \
  --algo ppo \
  --model "models/vanilla/${BASENAME}.zip" \
  --episodes 20
