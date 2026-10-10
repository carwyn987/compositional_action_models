#!/usr/bin/env bash
# Overnight diagnostic and preliminary-results runs, followed by a summary table.
#
#   ./scripts/overnight_runs.sh                       # defaults: ~8-9 h on 6 CPU cores (5 parallel runs)
#   JOBS=2 STEPS=500000 ./scripts/overnight_runs.sh   # fewer / shorter runs
#   DRY_RUN=1 ./scripts/overnight_runs.sh             # print the runs without starting them
#
# Groups (each run is one ./main.py training run with evaluation during training):
#   1. learnability   pickup alone, multi-hot, PPO and SAC, long budget: can the skill be learned at all,
#                     and which algorithm learns it faster? (debugging: if these fail, nothing else can work)
#   2. rq1            pickup + putdown, one shared PPO policy, every operator embedding condition, SEEDS
#                     seeds: one-hot, random, random-trainable, text (mock), multi-hot, multi-hot-trainable,
#                     compositional tree / slots / geometric. Preliminary RQ1 comparison and a check that
#                     every condition trains end to end.
#
# Throughput measured on 6 CPU cores: ~200 steps/s per PPO run with 3 in parallel (compositional slots /
# geometric ~120-200), SAC ~70; budgets: 18 rq1 runs x 1M + PPO 3M + SAC 1M steps.
#
# All runs use the shaped reward (sparse pickup reward is rarely discovered), 2 blocks, and the same
# evaluation settings, so their metrics.json files are comparable. A run whose log says it finished is
# skipped when the script is re-run, so an interrupted night can be continued by starting it again.
#
# Outputs, under OUT (default outputs/overnight_<date>):
#   <group>/<run name>/   the usual run directory (metrics.json, model.zip, tensorboard/, train.log, ...)
#   logs/<label>.log      each run's full output
#   failed.txt            labels of runs that exited with an error
#   summary.csv           final deterministic/stochastic success, AUC and steps to threshold per run and skill
# View the curves with: LOGDIR=<OUT> ./scripts/start_tensorboard.sh
#
# Environment variables (defaults in brackets):
#   OUT [outputs/overnight_<date>]  JOBS [cores - 1]  STEPS [1000000] steps per rq1 run
#   LONG_STEPS [3000000] PPO learnability steps (SAC gets a third: it is slower per step)
#   SEEDS ["0 1"]  EVAL_INTERVAL [50000]  EVAL_EPISODES [20]  SAVE_INTERVAL [100000]  DRY_RUN [0]

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUT="${OUT:-outputs/overnight_$(date +%Y%m%d)}"
JOBS="${JOBS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}"
STEPS="${STEPS:-1000000}"
LONG_STEPS="${LONG_STEPS:-3000000}"
SEEDS="${SEEDS:-0 1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON=".venv/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="python"

COMMON="--reward shaped --num-blocks 2 --eval-interval ${EVAL_INTERVAL} --eval-episodes ${EVAL_EPISODES} --save-interval ${SAVE_INTERVAL}"
RQ1_CONDITIONS=(
  "one-hot|--operator-encoder one-hot"
  "random|--operator-encoder random"
  "random-trainable|--operator-encoder random --trainable-operator-embedding"
  "text-mock|--operator-encoder text --text-backend mock --operator-text paragraph"
  "multi-hot|--operator-encoder multi-hot"
  "multi-hot-trainable|--operator-encoder multi-hot --trainable-operator-embedding"
  "compositional-tree|--operator-encoder compositional --compositional-architecture tree"
  "compositional-slots|--operator-encoder compositional --compositional-architecture slots"
  "compositional-geometric|--operator-encoder compositional --compositional-architecture geometric"
)

# One "label|group|arguments" line per run, longest runs first so parallel slots stay busy.
runs=(
  "learnability_ppo|learnability|--skills pickup --algorithm ppo --operator-encoder multi-hot --total-timesteps ${LONG_STEPS} --seed 0"
  "learnability_sac|learnability|--skills pickup --algorithm sac --operator-encoder multi-hot --total-timesteps $(( LONG_STEPS / 3 )) --seed 0"
)
for seed in ${SEEDS}; do
  for condition in "${RQ1_CONDITIONS[@]}"; do
    runs+=("rq1_${condition%%|*}_seed${seed}|rq1|--skills pickup putdown --algorithm ppo ${condition#*|} --total-timesteps ${STEPS} --seed ${seed}")
  done
done

mkdir -p "${OUT}/logs"
echo "overnight runs: ${#runs[@]} runs, ${JOBS} in parallel, output in ${OUT}"
commands="${OUT}/commands.txt"
: > "${commands}"
for run in "${runs[@]}"; do
  IFS='|' read -r label group arguments <<< "${run}"
  log="${OUT}/logs/${label}.log"
  if grep -q "training finished" "${log}" 2>/dev/null; then
    echo "  skip (finished): ${label}"
    continue
  fi
  command="${PYTHON} main.py ${arguments} ${COMMON} --output-dir ${OUT}/${group}"
  echo "  ${label}: ${command}"
  # One thread per run for torch / BLAS, so parallel runs do not oversubscribe the cores.
  echo "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${command} > ${log} 2>&1 || echo ${label} >> ${OUT}/failed.txt" >> "${commands}"
done
[[ "${DRY_RUN}" == "1" ]] && { echo "DRY_RUN=1: not starting"; exit 0; }

start=$(date +%s)
tr '\n' '\0' < "${commands}" | xargs -0 -P "${JOBS}" -I{} bash -c '{}'
echo "all runs done in $(( ($(date +%s) - start) / 60 )) min; failed: $(cat "${OUT}/failed.txt" 2>/dev/null | tr '\n' ' ')"

# Summary: one row per run, skill and evaluation mode, from each run's metrics.json.
"${PYTHON}" - "${OUT}" <<'EOF'
import csv, json, pathlib, sys
out = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(out.glob("*/*/metrics.json")):
    metrics = json.loads(path.read_text())
    last = metrics["curves"]["deterministic"]["points"][-1] if metrics["curves"]["deterministic"]["points"] else None
    for mode, skills in metrics["summary"].items():
        for skill, values in skills.items():
            rows.append({"group": path.parent.parent.name, "run": path.parent.name, "mode": mode, "skill": skill,
                         "last_step": last["step"] if last else None, **values})
if rows:
    with open(out / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{'run':58s} {'mode':13s} {'skill':8s} {'final':>6s} {'auc':>6s} {'steps_to_thr':>12s}")
    for row in rows:
        print(f"{row['run']:58s} {row['mode']:13s} {row['skill']:8s} {row['final_success_rate']:6.2f} "
              f"{row['auc']:6.2f} {str(row['steps_to_threshold']):>12s}")
    print(f"\nwrote {out / 'summary.csv'}")
EOF
