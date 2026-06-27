#!/usr/bin/env bash
set -euo pipefail

# Train + evaluate every Fetch skill with the VANILLA entrypoints, i.e. WITHOUT
# any symbolic action-model embedding in the observation. This is the no-embed
# control for the embedding experiments under scripts/embeddings/.
#
# Mirrors scripts/per_skill/train_all_and_eval.sh but drives
# experimental/train_vanilla_skill.py / experimental/rollout_vanilla_skill.py.
#
# Usage examples:
#   ./experimental/scripts/train_all_and_eval.sh
#   TIMESTEPS=200000 ALGO=sac EVAL_EPISODES=10 ./experimental/scripts/train_all_and_eval.sh
#   SKILLS="pickup pushleft pushright" ./experimental/scripts/train_all_and_eval.sh
#
# Environment variables:
#   ALGO           sac or ppo                 default: sac
#   TIMESTEPS      training steps per skill   default: 100000
#   SEED           base random seed           default: 0
#   EVAL_EPISODES  eval episodes per skill    default: 5
#   MODELS_DIR     model output directory     default: models/vanilla
#   LOGDIR         training/eval log directory default: runs/vanilla
#   SKILLS         space-separated skill list  default: all current skills
#   TEACHER        bc or off                  default: bc
#   TEACHER_ROLLOUTS        demos per skill   default: 8
#   TEACHER_GRADIENT_STEPS  BC updates        default: 500
#   TEACHER_BATCH_SIZE      BC batch size     default: 256
#   TEACHER_LR              BC learning rate  default: 1e-4

ALGO="${ALGO:-sac}"
TIMESTEPS="${TIMESTEPS:-100000}"
SEED="${SEED:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
MODELS_DIR="${MODELS_DIR:-models/vanilla}"
LOGDIR="${LOGDIR:-runs/vanilla}"
TEACHER="${TEACHER:-bc}"
TEACHER_ROLLOUTS="${TEACHER_ROLLOUTS:-8}"
TEACHER_GRADIENT_STEPS="${TEACHER_GRADIENT_STEPS:-500}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-256}"
TEACHER_LR="${TEACHER_LR:-1e-4}"

DEFAULT_SKILLS=(
  pickup
  putdown
  pushleft
  pushright
  pushforward
  pushbackward
  reach_top
)

if [[ -n "${SKILLS:-}" ]]; then
  # shellcheck disable=SC2206
  SKILL_LIST=(${SKILLS})
else
  SKILL_LIST=("${DEFAULT_SKILLS[@]}")
fi

case "${ALGO}" in
  sac|ppo) ;;
  *)
    echo "ERROR: ALGO must be 'sac' or 'ppo', got '${ALGO}'" >&2
    exit 2
    ;;
esac

case "${TEACHER}" in
  bc|off) ;;
  *)
    echo "ERROR: TEACHER must be 'bc' or 'off', got '${TEACHER}'" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p "${MODELS_DIR}" "${LOGDIR}/all_train" "${LOGDIR}/all_eval"

SUMMARY_FILE="${LOGDIR}/all_eval/summary_${ALGO}.txt"
: > "${SUMMARY_FILE}"

printf 'Training/evaluating VANILLA skills: %s\n' "${SKILL_LIST[*]}"
printf 'ALGO=%s TIMESTEPS=%s SEED=%s EVAL_EPISODES=%s TEACHER=%s\n' \
  "${ALGO}" "${TIMESTEPS}" "${SEED}" "${EVAL_EPISODES}" "${TEACHER}"

for SKILL in "${SKILL_LIST[@]}"; do
  echo
  echo "=============================="
  echo "Training skill: ${SKILL}"
  echo "=============================="

  # Single source of truth for the run name (tag 'vanilla' = no embedding),
  # so train, eval, logs, and the saved model all agree and never collide.
  BASENAME="$(python -m symb_model_embeddings.run_naming \
    --skill "${SKILL}" --algo "${ALGO}" --teacher "${TEACHER}" --seed "${SEED}" \
    --tag vanilla)"

  python experimental/train_vanilla_skill.py \
    --skill "${SKILL}" \
    --algo "${ALGO}" \
    --timesteps "${TIMESTEPS}" \
    --seed "${SEED}" \
    --models-dir "${MODELS_DIR}" \
    --logdir "${LOGDIR}" \
    --teacher "${TEACHER}" \
    --teacher-rollouts "${TEACHER_ROLLOUTS}" \
    --teacher-gradient-steps "${TEACHER_GRADIENT_STEPS}" \
    --teacher-batch-size "${TEACHER_BATCH_SIZE}" \
    --teacher-lr "${TEACHER_LR}" \
    2>&1 | tee "${LOGDIR}/all_train/${BASENAME}.log"

  MODEL_PATH="${MODELS_DIR}/${BASENAME}.zip"
  if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: expected model not found: ${MODEL_PATH}" >&2
    exit 1
  fi

  echo
  echo "=============================="
  echo "Evaluating skill: ${SKILL}"
  echo "=============================="

  {
    echo "skill=${SKILL} algo=${ALGO} model=${MODEL_PATH} episodes=${EVAL_EPISODES}"
    python experimental/rollout_vanilla_skill.py \
      --skill "${SKILL}" \
      --algo "${ALGO}" \
      --model "${MODEL_PATH}" \
      --episodes "${EVAL_EPISODES}" \
      --seed "$((SEED + 1000))" \
      --no-render
  } 2>&1 | tee "${LOGDIR}/all_eval/${BASENAME}.log" | tee -a "${SUMMARY_FILE}"

done

echo
echo "Done."
echo "Models: ${MODELS_DIR}/"
echo "Training logs: ${LOGDIR}/all_train/"
echo "Evaluation logs: ${LOGDIR}/all_eval/"
echo "Combined evaluation summary: ${SUMMARY_FILE}"
