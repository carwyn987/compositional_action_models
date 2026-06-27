#!/usr/bin/env bash
set -euo pipefail

# Vanilla (no symbolic embedding) pickup training + evaluation.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python experimental/train_vanilla_skill.py \
  --skill pickup \
  --algo ppo \
  --teacher bc \
  --teacher-rollouts 1000 \
  --teacher-gradient-steps 50000 \
  --teacher-batch-size 1024 \
  --teacher-lr 1e-4 \
  --timesteps 200000 \
  --models-dir models/vanilla

BASENAME="$(python -m symb_model_embeddings.run_naming --skill pickup --algo ppo --teacher bc --tag vanilla)"
MUJOCO_GL=glfw python experimental/rollout_vanilla_skill.py \
  --skill pickup \
  --algo ppo \
  --model "models/vanilla/${BASENAME}.zip" \
  --episodes 20
