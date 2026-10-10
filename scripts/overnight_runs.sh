#!/usr/bin/env bash
# Overnight runs for the research questions, followed by a summary table and a review report.
#
#   ./scripts/overnight_runs.sh                                   # every group, defaults below
#   RUN_GROUPS="learnability" ./scripts/overnight_runs.sh          # one group
#   CONDITIONS="one-hot compositional-tree" SEEDS="0 1" ./scripts/overnight_runs.sh
#   DRY_RUN=1 ./scripts/overnight_runs.sh                         # print the runs without starting them
#
# Groups (each run is one ./main.py training run with evaluation during training):
#   learnability  each skill alone (BASE_SKILLS and REPAIR_TO), multi-hot: can every skill be trained at
#                 all, and how fast? If one fails here, nothing using it can work.
#   rq1           RQ1: one shared policy on BASE_SKILLS per operator embedding condition (CONDITIONS) and
#                 seed: steps to threshold, AUC and final success per condition.
#   repair        RQ2 / RQ3: each rq1 run continued with REPAIR_FROM replaced by its repaired version
#                 REPAIR_TO (a copy of the rq1 run, resumed with --stop-skills REPAIR_TO): the repaired
#                 skill's zero-shot success at the resume step and its steps to threshold. One-hot and
#                 random also run with --operator-identity-aliases REPAIR_TO=REPAIR_FROM ("same-name"),
#                 where the repair is invisible to them; "new-name" gives the repaired skill its own code.
#                 Runs after its rq1 run in the same job; without the rq1 group it uses existing rq1 runs.
#
# Default skills: BASE_SKILLS = pickup putdown stack unstack pickup-raised place-beside. The repair of
# unstack is unstack-raised (unstack + effect (raised ?o)); raised is seen in training through
# pickup-raised, so a policy that reads the operator's structure can use it zero-shot.
#
# Measured SAC throughput with 6 skills, one thread per run: ~40 steps/s (one-hot, random, text,
# multi-hot), ~24 (compositional tree), ~23 (slots), ~18 (geometric). With 5 runs in parallel the
# defaults take ~9 h: each compositional rq1 + repair job ~8 h, each one-hot / random job (two repairs)
# ~5.5 h, each learnability run ~1 h. Scale the *_STEPS variables to the time available.
#
# All runs use the shaped reward, 2 blocks and the same evaluation settings, so their metrics.json
# files are comparable. A run whose log says it finished is skipped when the script is re-run, so an
# interrupted night can be continued by starting it again.
#
# Outputs, under OUT (default outputs/overnight_<date>):
#   learnability/<run name>/            the usual run directory (metrics.json, model.zip, tensorboard/, ...)
#   rq1/<condition>_seed<n>/<run name>/
#   repair/<condition>_<new-name|same-name>_seed<n>/   copy of the rq1 run, continued with the repaired skill
#   logs/<label>.log                    each run's full output
#   failed.txt                          labels of runs that exited with an error
#   summary.csv, review.txt             per run and skill metrics; scripts/review_runs.py's report
# View the curves with: LOGDIR=<OUT> ./scripts/start_tensorboard.sh
#
# Environment variables (defaults in brackets):
#   OUT [outputs/overnight_<date>]  JOBS [cores - 1]  RUN_GROUPS ["learnability rq1 repair"]
#   ALGORITHM [sac]  SEEDS ["0"]  CONDITIONS [one-hot random multi-hot-trainable text-mock
#   compositional-tree compositional-slots]  ALIAS_CONDITIONS ["one-hot random"]
#   BASE_SKILLS [see above]  REPAIR_FROM [unstack]  REPAIR_TO [unstack-raised]
#   LEARN_STEPS [150000]  RQ1_STEPS [500000]  REPAIR_STEPS [150000]
#   EVAL_INTERVAL [50000]  EVAL_EPISODES [20]  SAVE_INTERVAL [100000]  DRY_RUN [0]

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUT="${OUT:-outputs/overnight_$(date +%Y%m%d)}"
JOBS="${JOBS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}"
RUN_GROUPS="${RUN_GROUPS:-learnability rq1 repair}"
ALGORITHM="${ALGORITHM:-sac}"
SEEDS="${SEEDS:-0}"
CONDITIONS="${CONDITIONS:-one-hot random multi-hot-trainable text-mock compositional-tree compositional-slots}"
ALIAS_CONDITIONS="${ALIAS_CONDITIONS:-one-hot random}"
BASE_SKILLS="${BASE_SKILLS:-pickup putdown stack unstack pickup-raised place-beside}"
REPAIR_FROM="${REPAIR_FROM:-unstack}"
REPAIR_TO="${REPAIR_TO:-unstack-raised}"
LEARN_STEPS="${LEARN_STEPS:-150000}"
RQ1_STEPS="${RQ1_STEPS:-500000}"
REPAIR_STEPS="${REPAIR_STEPS:-150000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON=".venv/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="python"

# Operator embedding conditions: name -> main.py options.
declare -A CONDITION_OPTIONS=(
  [one-hot]="--operator-encoder one-hot"
  [random]="--operator-encoder random"
  [random-trainable]="--operator-encoder random --trainable-operator-embedding"
  [text-mock]="--operator-encoder text --text-backend mock --operator-text paragraph"
  [multi-hot]="--operator-encoder multi-hot"
  [multi-hot-trainable]="--operator-encoder multi-hot --trainable-operator-embedding"
  [compositional-tree]="--operator-encoder compositional --compositional-architecture tree"
  [compositional-slots]="--operator-encoder compositional --compositional-architecture slots"
  [compositional-geometric]="--operator-encoder compositional --compositional-architecture geometric"
)
for condition in ${CONDITIONS}; do
  [[ -n "${CONDITION_OPTIONS[${condition}]:-}" ]] || { echo "unknown condition ${condition}; known: ${!CONDITION_OPTIONS[*]}"; exit 1; }
done
in_groups() { [[ " ${RUN_GROUPS} " == *" $1 "* ]]; }
REPAIRED_SKILLS="${BASE_SKILLS/${REPAIR_FROM}/${REPAIR_TO}}"  # REPAIR_FROM replaced in place

COMMON="--algorithm ${ALGORITHM} --reward shaped --num-blocks 2 --eval-interval ${EVAL_INTERVAL} --eval-episodes ${EVAL_EPISODES} --save-interval ${SAVE_INTERVAL}"
ENVIRONMENT="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"  # one thread per run, so parallel runs do not oversubscribe

mkdir -p "${OUT}/logs"
commands="${OUT}/commands.txt"
: > "${commands}"

# step <label> <command>: one shell line that runs command, logging to logs/<label>.log, unless that log
# already says it finished; a failure is recorded in failed.txt and stops the rest of its job.
step() {
  local label="$1" command="$2" log="${OUT}/logs/$1.log"
  echo "{ grep -q 'training finished' ${log} 2>/dev/null || { ${ENVIRONMENT} ${command} > ${log} 2>&1 || { echo ${label} >> ${OUT}/failed.txt; false; }; }; }"
}

jobs_listed=0
add_job() { echo "$1" >> "${commands}"; jobs_listed=$(( jobs_listed + 1 )); }

# rq1 (+ its repairs) first: they are the longest jobs, so the parallel slots stay busy.
for seed in ${SEEDS}; do
  for condition in ${CONDITIONS}; do
    rq1_output="${OUT}/rq1/${condition}_seed${seed}"
    job=""
    if in_groups rq1; then
      job="$(step "rq1_${condition}_seed${seed}" "${PYTHON} main.py --skills ${BASE_SKILLS} ${CONDITION_OPTIONS[${condition}]} --total-timesteps ${RQ1_STEPS} --seed ${seed} ${COMMON} --output-dir ${rq1_output}")"
    fi
    if in_groups repair; then
      variants="new-name"
      [[ " ${ALIAS_CONDITIONS} " == *" ${condition} "* ]] && variants+=" same-name"
      for variant in ${variants}; do
        repair_run="${OUT}/repair/${condition}_${variant}_seed${seed}"
        aliases=""
        [[ "${variant}" == "same-name" ]] && aliases="--operator-identity-aliases ${REPAIR_TO}=${REPAIR_FROM}"
        # copy the finished rq1 run (once), then resume the copy with the repaired skill
        copy="{ [ -d ${repair_run} ] || { mkdir -p ${OUT}/repair && cp -r \$(ls -d ${rq1_output}/*/ | head -1) ${repair_run}; }; }"
        resume="$(step "repair_${condition}_${variant}_seed${seed}" "${PYTHON} main.py --run-directory ${repair_run} --skills ${REPAIRED_SKILLS} --stop-skills ${REPAIR_TO} --total-timesteps ${REPAIR_STEPS} ${aliases}")"
        job="${job:+${job} && }${copy} && ${resume}"
      done
    fi
    [[ -n "${job}" ]] && add_job "${job}"
  done
done
if in_groups learnability; then
  for skill in ${BASE_SKILLS} ${REPAIR_TO}; do
    add_job "$(step "learnability_${skill}" "${PYTHON} main.py --skills ${skill} --operator-encoder multi-hot --total-timesteps ${LEARN_STEPS} --seed 0 ${COMMON} --output-dir ${OUT}/learnability")"
  done
fi

echo "overnight runs: ${jobs_listed} jobs (groups: ${RUN_GROUPS}; ${ALGORITHM}), ${JOBS} in parallel, output in ${OUT}"
echo "  base skills: ${BASE_SKILLS}"
in_groups repair && echo "  repair: ${REPAIR_FROM} -> ${REPAIR_TO}; repaired skill set: ${REPAIRED_SKILLS}"
[[ "${DRY_RUN}" == "1" ]] && { cat "${commands}"; echo "DRY_RUN=1: not starting"; exit 0; }

start=$(date +%s)
tr '\n' '\0' < "${commands}" | xargs -0 -P "${JOBS}" -I{} bash -c '{}'
echo "all runs done in $(( ($(date +%s) - start) / 60 )) min; failed: $(cat "${OUT}/failed.txt" 2>/dev/null | tr '\n' ' ')"

# Summary: one row per run, skill and evaluation mode, from each run's metrics.json.
"${PYTHON}" - "${OUT}" <<'EOF'
import csv, json, pathlib, sys
out = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(out.glob("**/metrics.json")):
    metrics = json.loads(path.read_text())
    points = metrics["curves"]["deterministic"]["points"]
    for mode, skills in metrics["summary"].items():
        for skill, values in skills.items():
            rows.append({"group": path.relative_to(out).parts[0], "run": path.parent.name, "mode": mode,
                         "skill": skill, "last_step": points[-1]["step"] if points else None, **values})
if rows:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(out / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out / 'summary.csv'}")
EOF
"${PYTHON}" scripts/review_runs.py "${OUT}" > /dev/null && echo "wrote ${OUT}/review.txt"
