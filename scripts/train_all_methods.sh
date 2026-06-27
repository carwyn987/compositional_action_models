#!/usr/bin/env bash
set -euo pipefail

# Train a SINGLE shared model on several skills at once (default: pickup +
# putdown, configurable via SKILLS), where the only thing that differentiates
# the skills to the policy is the symbolic embedding vector. This is repeated
# once per embedding scheme AND per embedding mode (frozen vs trainable), all
# sharing the EXACT same hyperparameters, so they can be compared head-to-head:
#
#   schemes: mock (random per-skill vector)
#            name (text embedding of the operator name)
#            full (text embedding of the full PDDL action-model string)
#   modes:   frozen    -- embedding held fixed
#            trainable -- embedding is a policy parameter, learned during RL
#
# => six trainings by default (3 schemes x 2 modes). Trainable runs additionally
# record how the embeddings move over training to EMBED_LOG_DIR; visualise with
#   python plot_embeddings.py
#
# Vanilla (no embedding) is intentionally excluded: with no embedding the shared
# policy gets no signal telling the skills apart, so multi-skill training is
# impossible without one. The learned embeddings are never written back to the
# pretrained embedding source -- only into the saved policy and the log.
#
# Thanks to the run-naming scheme, all runs save to distinct files.
#
# Usage:
#   ./scripts/train_all_methods.sh
#   TEACHER_ROLLOUTS=20 TIMESTEPS=200000 TEACHER_LR=3e-4 ./scripts/train_all_methods.sh
#   SKILLS="pickup putdown pushleft" ALGO=ppo ./scripts/train_all_methods.sh
#   MODES=trainable ./scripts/train_all_methods.sh        # only the trainable runs
#
# Environment variables (shared across all trainings):
#   SKILLS                  space-separated skills      default: "pickup putdown"
#   MODES                   "frozen trainable"          default: "frozen trainable"
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
#   EMBED_LOG_DIR           embedding-trajectory logs   default: embedding_logs

SKILLS="${SKILLS:-pickup putdown}"
MODES="${MODES:-frozen trainable}"
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
EMBED_LOG_DIR="${EMBED_LOG_DIR:-embedding_logs}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC2206
SKILL_LIST=(${SKILLS})
# shellcheck disable=SC2206
MODE_LIST=(${MODES})

# Hyperparameters every training shares (the embedding scheme/mode flags are
# appended per-run below).
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
  --embed-log-dir "${EMBED_LOG_DIR}"
)

# Embedding scheme name -> the train_multiskill.py flags that select it.
SCHEME_NAMES=(mock name full)
declare -A SCHEME_FLAGS=(
  [mock]="--embedder mock"
  [name]="--embedder text --embed-source name --embed-backend ${EMBED_BACKEND}"
  [full]="--embedder text --embed-source action_model --embed-backend ${EMBED_BACKEND}"
)

printf 'Training one shared model on skills [%s] (algo=%s seed=%s timesteps=%s)\n' \
  "${SKILL_LIST[*]}" "${ALGO}" "${SEED}" "${TIMESTEPS}"
printf '  schemes=[%s] modes=[%s]\n' "${SCHEME_NAMES[*]}" "${MODE_LIST[*]}"
printf '  teacher=%s rollouts=%s grad_steps=%s batch=%s lr=%s sampling=%s\n' \
  "${TEACHER}" "${TEACHER_ROLLOUTS}" "${TEACHER_GRADIENT_STEPS}" \
  "${TEACHER_BATCH_SIZE}" "${TEACHER_LR}" "${SAMPLING}"
printf '  embed_backend=%s embed_size=%s\n' "${EMBED_BACKEND}" "${EMBED_SIZE}"

for MODE in "${MODE_LIST[@]}"; do
  case "${MODE}" in
    frozen)    TRAINABLE=false ;;
    trainable) TRAINABLE=true ;;
    *) echo "ERROR: MODE must be 'frozen' or 'trainable', got '${MODE}'" >&2; exit 2 ;;
  esac
  for SCHEME in "${SCHEME_NAMES[@]}"; do
    echo
    echo "=== ${SCHEME} embedding [${MODE}] ==="
    # shellcheck disable=SC2086
    python train_multiskill.py "${COMMON[@]}" \
      ${SCHEME_FLAGS[$SCHEME]} \
      --embed-size "${EMBED_SIZE}" \
      --embed-trainable "${TRAINABLE}"
  done
done

echo
echo "Done. Models in ${MODELS_DIR}/, TB logs in ${LOGDIR}/, embedding logs in ${EMBED_LOG_DIR}/."
echo "View training curves with: ./scripts/tensorboard.sh"
echo "Plot embedding movement with: python plot_embeddings.py --log-dir ${EMBED_LOG_DIR}"
