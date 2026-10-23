#!/usr/bin/env bash
# A sample repair scenario (PPO by default), small enough for a few hours: can a policy carry out an operator it was never
# trained on, from the operator's structure?
#
#   ./scripts/sample_repair_scenario.sh                 # every step; outputs in outputs/sample_repair_<date>/
#   ./scripts/sample_repair_scenario.sh evaluate        # one step: check, train, repair or evaluate
#   STEPS=2000000 ./scripts/sample_repair_scenario.sh   # a longer training run (see the settings below)
#
# Steps
#   check     (seconds) a constant close-and-lift from unstack's setup (no retreat) must not count as unstack-raised: lifting on and
#             on through the raised band is not stopping in it (tests/cam/environments/test_settled_success.py)
#   train     three PPO runs side by side:
#               unstack    one-hot, unstack alone, UNSTACK_STEPS
#               structure  compositional-tree on pickup, unstack and unstack-raised, STEPS: the pair unstack /
#                          unstack-raised shows the policy what (raised ?o) asks for, in the unstack context
#               scratch    compositional-tree on pickup-raised alone, REPAIR_STEPS: the from-scratch baseline
#   repair    the structure policy continued with pickup-raised added, training only it (--repair-mode isolated)
#             until it holds the success threshold, at most REPAIR_STEPS; pickup-raised is evaluated where it is
#             added (zero-shot) and 1k, 2k, 4k, ... steps later
#   evaluate  the unstack policy on unstack-raised given unstack's exact embedding (--operator-identity-aliases): it
#             must fail, so training on unstack alone does not give unstack-raised. Whether the structure run
#             learned its three skills (the repair means nothing otherwise); pickup-raised's zero-shot success and
#             steps to threshold in the repair against from scratch (review_runs.py sections 2 and 5); the
#             wrong-embedding control (pickup-raised given pickup's operator) before and after the repair; and
#             whether each policy is stuck or slowly getting there
#
# The compositional runs leave the operator name out (--compositional-name none): "pickup-raised" shares the word
# "raised" with "unstack-raised", so a name embedding would carry the repair by itself.
#
# Settings (environment variables, defaults in brackets): OUT [outputs/sample_repair_<date>]  STEPS [1000000]
# UNSTACK_STEPS [300000]  REPAIR_STEPS [500000]  SEED [0]  ALGORITHM [ppo; mtsac explores more precisely near
# the blocks, but runs about 4x slower per step]  DEVICE [cpu: PPO's small networks run faster there]
# EVAL_EPISODES [20]  THREADS [cores / 3] torch threads per run  PYTHON [.venv/bin/python]. Finished runs are
# skipped when run again.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
OUT=${OUT:-outputs/sample_repair_$(date +%Y%m%d)}
STEPS=${STEPS:-1000000}
UNSTACK_STEPS=${UNSTACK_STEPS:-300000}
REPAIR_STEPS=${REPAIR_STEPS:-500000}
SEED=${SEED:-0}
DEVICE=${DEVICE:-cpu}
ALGORITHM=${ALGORITHM:-ppo}
EVAL_EPISODES=${EVAL_EPISODES:-20}
THREADS=${THREADS:-$(( $(nproc) / 3 > 0 ? $(nproc) / 3 : 1 ))}
export OMP_NUM_THREADS=$THREADS  # no CPU oversubscription with three runs side by side
# this repository's code only: not a PYTHONPATH from the shell (e.g. a sourced ROS setup, whose pytest plugins fail)
export PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
COMMON=(--algorithm "$ALGORITHM" --reward shaped --num-blocks 2 --skill-sampling competence --seed "$SEED" --device "$DEVICE"
        --eval-episodes "$EVAL_EPISODES")
TREE=(--operator-encoder compositional --compositional-architecture tree --compositional-name none)
mkdir -p "$OUT/logs"

run_of() { ls -d "$OUT/$1"/*/ 2>/dev/null | head -1; }  # the run directory main.py made under OUT/<label>
finished() { [ -f "$OUT/logs/$1.log" ] && grep -q "training finished" "$OUT/logs/$1.log"; }

train() {  # label, then main.py options: a new run under OUT/<label>, logged to OUT/logs/<label>.log
    local label=$1; shift
    if finished "$label"; then echo "  $label: finished already ($(run_of "$label"))"; return 0; fi
    echo "  $label: training, log in $OUT/logs/$label.log"
    "$PYTHON" main.py "$@" "${COMMON[@]}" --output-dir "$OUT/$label" > "$OUT/logs/$label.log" 2>&1
    echo "  $label: done ($(grep -o 'training finished at step [0-9]*' "$OUT/logs/$label.log" | tail -1 || true))"
}

step_check() {
    echo "== check: lifting on and on through the raised band is not unstack-raised"
    "$PYTHON" -m pytest -q tests/cam/environments/test_settled_success.py
}

step_train() {
    echo "== train ($ALGORITHM, side by side; PPO: about 2-3 hours at 1M steps on 6 CPU cores)"
    local pids=() failed=0
    train unstack --skills unstack --operator-encoder one-hot --total-timesteps "$UNSTACK_STEPS" --eval-interval 50000 &
    pids+=($!)
    train structure --skills pickup unstack unstack-raised "${TREE[@]}" --total-timesteps "$STEPS" \
        --eval-interval 50000 &
    pids+=($!)
    train scratch --skills pickup-raised "${TREE[@]}" --total-timesteps "$REPAIR_STEPS" --eval-interval 20000 &
    pids+=($!)
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [ "$failed" = 0 ] || { echo "a training run failed: see $OUT/logs/"; exit 1; }
}

step_repair() {
    echo "== repair: the structure policy, with pickup-raised added and trained alone"
    local structure; structure=$(run_of structure)
    [ -n "$structure" ] && finished structure || { echo "train the structure run first"; exit 1; }
    if [ -f "$OUT/logs/repair.log" ] && grep -q "training finished" "$OUT/logs/repair.log"; then
        echo "  repair: finished already"; return 0
    fi
    rm -rf "$OUT/repair"
    mkdir -p "$OUT/repair"
    cp -r "$structure" "$OUT/repair/"
    local run; run=$(run_of repair)
    echo "  repair: training, log in $OUT/logs/repair.log"
    "$PYTHON" main.py --run-directory "$run" --skills pickup pickup-raised unstack unstack-raised \
        --stop-skills pickup-raised --repair-mode isolated --total-timesteps "$REPAIR_STEPS" --eval-interval 10000 \
        --device "$DEVICE" > "$OUT/logs/repair.log" 2>&1
    echo "  repair: done ($(grep -o 'training finished at step [0-9]*' "$OUT/logs/repair.log" | tail -1 || true))"
}

evaluate_skills() {  # run directory, then main.py --mode evaluate options; prints each skill's success
    local run=$1; shift
    "$PYTHON" main.py --mode evaluate --run-directory "$run" --eval-episodes 50 --device cpu "$@" 2>&1 \
        | grep -o "evaluation at step .*: success [0-9.]* ± [0-9.]*" | sed 's/^/    /' || true
}

step_evaluate() {
    local unstack structure repair scratch
    unstack=$(run_of unstack); structure=$(run_of structure); repair=$(run_of repair); scratch=$(run_of scratch)

    echo "== 1. training on unstack alone does not give unstack-raised (same embedding: expect unstack high, unstack-raised ~0)"
    [ -n "$unstack" ] && evaluate_skills "$unstack" --eval-skills unstack unstack-raised \
        --operator-identity-aliases unstack-raised=unstack

    echo "== 2. the repair: pickup, unstack, unstack-raised -> pickup-raised (compositional-tree)"
    "$PYTHON" scripts/review_runs.py "$OUT" --no-embeddings > /dev/null
    echo "-- did the structure run learn its skills? (final success; the repair needs all three)"
    sed -n '/^2\./,/^3\./p' "$OUT/review.txt" | grep -A6 "^   structure/" | grep "deter" || true
    echo "-- pickup-raised: zero-shot where it is added, and steps to threshold (repair; then from scratch)"
    sed -n '/^5\./,/^6\./p' "$OUT/review.txt" | grep -A3 "pickup-raised" || true
    sed -n '/^2\./,/^3\./p' "$OUT/review.txt" | grep -A2 "^   scratch/" | grep "deter" || true
    echo "-- the wrong-embedding control: pickup-raised given pickup's operator (no change: the operator is not read)"
    for run in "$structure" "$repair"; do
        [ -n "$run" ] || continue
        echo "   $run"
        "$PYTHON" scripts/wrong_embedding_control.py "$run" --skill pickup-raised --as pickup --episodes 20 2>/dev/null \
            | grep -v Adroit | sed 's/^/    /' || true
    done
    echo "-- final success, 50 episodes per skill"
    [ -n "$repair" ] && evaluate_skills "$repair" --eval-skills pickup pickup-raised unstack unstack-raised
    [ -n "$scratch" ] && evaluate_skills "$scratch" --eval-skills pickup-raised
    echo "-- stuck, or slowly getting there?"
    [ -n "$repair" ] && "$PYTHON" scripts/check_policy_progress.py "$repair" --skills pickup-raised pickup \
        --episodes 5 2>/dev/null | grep -v Adroit || true
    echo
    echo "Full report: $OUT/review.txt   Watch: ./scripts/watch_best_policies.py --folder $OUT"
}

case "${1:-all}" in
    check) step_check ;;
    train) step_train ;;
    repair) step_repair ;;
    evaluate) step_evaluate ;;
    all) step_check; step_train; step_repair; step_evaluate ;;
    *) echo "usage: $0 [check|train|repair|evaluate|all]"; exit 2 ;;
esac
