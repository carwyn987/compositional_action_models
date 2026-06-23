#!/bin/bash

python experimental/train_vanilla_skill.py \
  --skill putdown \
  --algo ppo \
  --teacher bc \
  --teacher-rollouts 500 \
  --teacher-gradient-steps 20000 \
  --teacher-batch-size 1024 \
  --teacher-lr 1e-4 \
  --timesteps 100000 \
  --models-dir models

MUJOCO_GL=glfw python experimental/rollout_vanilla_skill.py \
  --skill putdown \
  --algo ppo \
  --model models/putdown_ppo.zip \
  --episodes 20
