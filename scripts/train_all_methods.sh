#!/usr/bin/env bash
set -euo pipefail

# Train one skill (default: pickup) four ways for a head-to-head comparison of
# the symbolic-embedding schemes, all sharing the EXACT same hyperparameters so
# the embedding is the only thing that differs:
#
#   1. vanilla  -- no embedding in the observation (control)
#   2. mock     -- fixed random per-skill vector
#   3. name     -- pretrained text embedding of the operator name
#   4. full     -- pretrained text embedding of the full PDDL action-model string
#
# Thanks to the run-naming scheme, the four runs save to distinct files and
# never overwrite each other.
#
# Usage:
#   ./scripts/train_all_methods.sh
#   TEACHER_ROLLOUTS=20 TIMESTEPS=200000 TEACHER_LR=3e-4 ./scripts/train_all_methods.sh
#   SKILL=stack ALGO=ppo ./scripts/train_all_methods.sh
#
# Environment variables (shared across all four trainings):
#   SKILL                   skill to train             default: pickup
#   ALGO                    sac or ppo                 default: ppo
#   SEED                    random seed                default: 0
#   TIMESTEPS               RL training steps          default: 100000
#   TEACHER                 bc or off                  default: bc
#   TEACHER_ROLLOUTS        teacher demos (BC episodes) default: 5
#   TEACHER_GRADIENT_STEPS  BC update steps            default: 500
#   TEACHER_BATCH_SIZE      BC batch size              default: 256
#   TEACHER_LR              BC learning rate           default: 1e-4
#   EMBED_BACKEND           hash (offline) or openai   default: hash
#   EMBED_SIZE              embedding dimensionality    default: 32
#   MODELS_DIR              embedding model output dir  default: models
#   VANILLA_MODELS_DIR      vanilla model output dir    default: models/vanilla
#   LOGDIR                  TensorBoard/monitor logs    default: runs

SKILL="${SKILL:-pickup}"
ALGO="${ALGO:-ppo}"
SEED="${SEED:-0}"
TIMESTEPS="${TIMESTEPS:-100000}"
TEACHER="${TEACHER:-bc}"
TEACHER_ROLLOUTS="${TEACHER_ROLLOUTS:-5}"
TEACHER_GRADIENT_STEPS="${TEACHER_GRADIENT_STEPS:-500}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-256}"
TEACHER_LR="${TEACHER_LR:-1e-4}"
EMBED_BACKEND="${EMBED_BACKEND:-hash}"
EMBED_SIZE="${EMBED_SIZE:-32}"
MODELS_DIR="${MODELS_DIR:-models}"
VANILLA_MODELS_DIR="${VANILLA_MODELS_DIR:-models/vanilla}"
LOGDIR="${LOGDIR:-runs}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Hyperparameters every training shares.
COMMON=(
  --skill "${SKILL}"
  --algo "${ALGO}"
  --seed "${SEED}"
  --timesteps "${TIMESTEPS}"
  --teacher "${TEACHER}"
  --teacher-rollouts "${TEACHER_ROLLOUTS}"
  --teacher-gradient-steps "${TEACHER_GRADIENT_STEPS}"
  --teacher-batch-size "${TEACHER_BATCH_SIZE}"
  --teacher-lr "${TEACHER_LR}"
  --logdir "${LOGDIR}"
)

printf 'Training "%s" (algo=%s seed=%s timesteps=%s) four ways:\n' \
  "${SKILL}" "${ALGO}" "${SEED}" "${TIMESTEPS}"
printf '  teacher=%s rollouts=%s grad_steps=%s batch=%s lr=%s\n' \
  "${TEACHER}" "${TEACHER_ROLLOUTS}" "${TEACHER_GRADIENT_STEPS}" \
  "${TEACHER_BATCH_SIZE}" "${TEACHER_LR}"
printf '  embed_backend=%s embed_size=%s\n' "${EMBED_BACKEND}" "${EMBED_SIZE}"

echo
echo "=== [1/4] vanilla (no embedding) ==="
python experimental/train_vanilla_skill.py "${COMMON[@]}" \
  --models-dir "${VANILLA_MODELS_DIR}"

echo
echo "=== [2/4] mock embedding (random per-skill vector) ==="
python train_skill.py "${COMMON[@]}" \
  --models-dir "${MODELS_DIR}" \
  --embedder mock --embed-size "${EMBED_SIZE}"

echo
echo "=== [3/4] name embedding (text embedding of operator name) ==="
python train_skill.py "${COMMON[@]}" \
  --models-dir "${MODELS_DIR}" \
  --embedder text --embed-source name \
  --embed-backend "${EMBED_BACKEND}" --embed-size "${EMBED_SIZE}"

echo
echo "=== [4/4] full embedding (text embedding of full action-model string) ==="
python train_skill.py "${COMMON[@]}" \
  --models-dir "${MODELS_DIR}" \
  --embedder text --embed-source action_model \
  --embed-backend "${EMBED_BACKEND}" --embed-size "${EMBED_SIZE}"

echo
echo "Done. Models in ${MODELS_DIR}/ and ${VANILLA_MODELS_DIR}/, logs in ${LOGDIR}/."
echo "View training curves with: ./scripts/tensorboard.sh"
