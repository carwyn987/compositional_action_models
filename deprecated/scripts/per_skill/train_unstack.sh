#!/bin/bash

# Train the unstack skill with PPO + behavior-cloning warm-start, then evaluate.

python train_skill.py \
  --skill unstack \
  --algo ppo \
  --teacher bc \
  --teacher-rollouts 1000 \
  --teacher-gradient-steps 50000 \
  --teacher-batch-size 1024 \
  --teacher-lr 1e-4 \
  --timesteps 300000 \
  --models-dir models

# Derive the same tagged name train_skill.py just saved (default mock embedding).
BASENAME="$(python -m symb_model_embeddings.run_naming --skill unstack --algo ppo --teacher bc)"
MUJOCO_GL=glfw python rollout_skill.py \
  --skill unstack \
  --algo ppo \
  --model "models/${BASENAME}.zip" \
  --episodes 20
