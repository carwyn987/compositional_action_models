#!/usr/bin/env bash
set -euo pipefail

# Train a SINGLE shared model on several skills at once (default: pickup +
# putdown, configurable via SKILLS), where the only thing that differentiates
# the skills to the policy is the symbolic embedding vector. This is repeated
# once per embedding scheme so they can be compared head-to-head, all sharing
# the EXACT same hyperparameters:
#
#   1. mock  -- random (but per-skill distinct) vector
#   2. name  -- pretrained text embedding of the operator name
#   3. full  -- pretrained text embedding of the full PDDL action-model string
#
# Vanilla (no embedding) is intentionally excluded: with no embedding the shared
# policy gets no signal telling the skills apart, so multi-skill training is
# impossible without one.
#
# Thanks to the run-naming scheme, the three runs save to distinct files and
# never overwrite each other.
#
# Usage:
#   ./scripts/train_all_methods.sh
#   TEACHER_ROLLOUTS=20 TIMESTEPS=200000 TEACHER_LR=3e-4 ./scripts/train_all_methods.sh
#   SKILLS="pickup putdown pushleft" ALGO=ppo ./scripts/train_all_methods.sh
#
# Environment variables (shared across all three trainings):
#   SKILLS                  space-separated skills      default: "pickup putdown"
#   ALGO                    sac or ppo                 default: ppo
#   SEED                    random seed                default: 0
#   TIMESTEPS               RL training steps          default: 100000
#   SAMPLING                random or round_robin      default: random
#   TEACHER                 bc or off                  default: bc
#   TEACHER_ROLLOUTS        teacher demos per skill    default: 5
#   TEACHER_GRADIENT_STEPS  BC update steps            default: 500
#   TEACHER_BATCH_SIZE      BC batch size              default: 256
#   TEACHER_LR              BC learning rate           default: 1e-4
#   EMBED_BACKEND           hash (offline) or openai   default: hash
#   EMBED_SIZE              embedding dimensionality    default: 32
#   MODELS_DIR              model output directory      default: models
#   LOGDIR                  TensorBoard/monitor logs    default: runs

SKILLS="${SKILLS:-pickup putdown}"
ALGO="${ALGO:-ppo}"
SEED="${SEED:-0}"
TIMESTEPS="${TIMESTEPS:-100000}"
SAMPLING="${SAMPLING:-random}"
TEACHER="${TEACHER:-bc}"
TEACHER_ROLLOUTS="${TEACHER_ROLLOUTS:-5}"
TEACHER_GRADIENT_STEPS="${TEACHER_GRADIENT_STEPS:-500}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-256}"
TEACHER_LR="${TEACHER_LR:-1e-4}"
EMBED_BACKEND="${EMBED_BACKEND:-hash}"
EMBED_SIZE="${EMBED_SIZE:-32}"
MODELS_DIR="${MODELS_DIR:-models}"
LOGDIR="${LOGDIR:-runs}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC2206
SKILL_LIST=(${SKILLS})

# Hyperparameters every training shares (the embedding flags are appended
# per-method below).
COMMON=(
  --skills "${SKILL_LIST[@]}"
  --algo "${ALGO}"
  --seed "${SEED}"
  --timesteps "${TIMESTEPS}"
  --sampling "${SAMPLING}"
  --teacher "${TEACHER}"
  --teacher-rollouts "${TEACHER_ROLLOUTS}"
  --teacher-gradient-steps "${TEACHER_GRADIENT_STEPS}"
  --teacher-batch-size "${TEACHER_BATCH_SIZE}"
  --teacher-lr "${TEACHER_LR}"
  --logdir "${LOGDIR}"
  --models-dir "${MODELS_DIR}"
)

printf 'Training one shared model on skills [%s] (algo=%s seed=%s timesteps=%s) three ways:\n' \
  "${SKILL_LIST[*]}" "${ALGO}" "${SEED}" "${TIMESTEPS}"
printf '  teacher=%s rollouts=%s grad_steps=%s batch=%s lr=%s sampling=%s\n' \
  "${TEACHER}" "${TEACHER_ROLLOUTS}" "${TEACHER_GRADIENT_STEPS}" \
  "${TEACHER_BATCH_SIZE}" "${TEACHER_LR}" "${SAMPLING}"
printf '  embed_backend=%s embed_size=%s\n' "${EMBED_BACKEND}" "${EMBED_SIZE}"

echo
echo "=== [1/3] mock embedding (random per-skill vector) ==="
python train_multiskill.py "${COMMON[@]}" \
  --embedder mock --embed-size "${EMBED_SIZE}"

echo
echo "=== [2/3] name embedding (text embedding of operator name) ==="
python train_multiskill.py "${COMMON[@]}" \
  --embedder text --embed-source name \
  --embed-backend "${EMBED_BACKEND}" --embed-size "${EMBED_SIZE}"

echo
echo "=== [3/3] full embedding (text embedding of full action-model string) ==="
python train_multiskill.py "${COMMON[@]}" \
  --embedder text --embed-source action_model \
  --embed-backend "${EMBED_BACKEND}" --embed-size "${EMBED_SIZE}"

echo
echo "Done. Models in ${MODELS_DIR}/, logs in ${LOGDIR}/."
echo "View training curves with: ./scripts/tensorboard.sh"
