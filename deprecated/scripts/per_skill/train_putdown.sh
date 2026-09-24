#!/bin/bash

python train_skill.py \
  --skill putdown \
  --algo ppo \
  --teacher bc \
  --teacher-rollouts 500 \
  --teacher-gradient-steps 20000 \
  --teacher-batch-size 1024 \
  --teacher-lr 1e-4 \
  --timesteps 100000 \
  --models-dir models

# Derive the same tagged name train_skill.py just saved (default mock embedding).
BASENAME="$(python -m symb_model_embeddings.run_naming --skill putdown --algo ppo --teacher bc)"
MUJOCO_GL=glfw python rollout_skill.py \
  --skill putdown \
  --algo ppo \
  --model "models/${BASENAME}.zip" \
  --episodes 20
