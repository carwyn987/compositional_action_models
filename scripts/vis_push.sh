#!/bin/bash

for SKILL in pushleft pushright pushforward pushbackward; do

  MUJOCO_GL=glfw python rollout_skill.py \
    --skill "${SKILL}" \
    --algo ppo \
    --model "models/${SKILL}_ppo.zip" \
    --episodes 5

done
