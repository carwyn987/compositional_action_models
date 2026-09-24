#!/bin/bash

for SKILL in pushleft pushright pushforward pushbackward; do

  # Match the tagged name train_push.sh saved (default mock embedding).
  BASENAME="$(python -m symb_model_embeddings.run_naming --skill "${SKILL}" --algo ppo --teacher bc)"
  MUJOCO_GL=glfw python rollout_skill.py \
    --skill "${SKILL}" \
    --algo ppo \
    --model "models/${BASENAME}.zip" \
    --episodes 5

done
