#!/usr/bin/env bash
set -euo pipefail

# Vanilla (no symbolic embedding) training of the four directional push skills.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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
    --models-dir models/vanilla
done
