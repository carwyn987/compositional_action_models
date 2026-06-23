#!/bin/bash

# Train the stack skill with PPO + behavior-cloning warm-start, then evaluate.

python experimental/train_vanilla_skill.py \
  --skill stack \
  --algo ppo \
  --teacher bc \
  --teacher-rollouts 5000 \
  --teacher-gradient-steps 200000 \
  --teacher-batch-size 1024 \
  --teacher-lr 1e-4 \
  --timesteps 1000000 \
  --models-dir models

MUJOCO_GL=glfw python experimental/rollout_vanilla_skill.py \
  --skill stack \
  --algo ppo \
  --model models/stack_ppo.zip \
  --episodes 20
