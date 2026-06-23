#!/bin/bash

for SKILL in pushleft pushright pushforward pushbackward; do

  MUJOCO_GL=glfw python experimental/rollout_vanilla_skill.py \
    --skill "${SKILL}" \
    --algo ppo \
    --model "models/${SKILL}_ppo.zip" \
    --episodes 5

done
