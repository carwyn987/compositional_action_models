#!/bin/bash

# Train the four directional push skills independently with PPO.
# Mirrors train_pickup.sh but loops over every push direction and does not
# play/evaluate the trained policies.

for SKILL in pushleft pushright pushforward pushbackward; do
  python experimental/train_vanilla_skill.py \
    --skill "${SKILL}" \
    --algo ppo \
    --teacher bc \
    --teacher-rollouts 1000 \
    --teacher-gradient-steps 50000 \
    --teacher-batch-size 1024 \
    --teacher-lr 1e-4 \
    --timesteps 200000 \
    --models-dir models
done
