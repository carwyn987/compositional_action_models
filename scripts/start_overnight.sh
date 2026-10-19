#!/usr/bin/env bash
# Start an overnight experiment (scripts/overnight_runs.py) from the current branch, in this terminal.
#
#   ./scripts/start_overnight.sh --experiment rq3-repair-minimal   # check, then run its plan (2 runs at a time), explaining each
#   DEVICE=cuda ./scripts/start_overnight.sh --experiment rq1    # on a GPU (fails fast if CUDA is not available)
#   JOBS=1 ./scripts/start_overnight.sh --experiment rq1         # one run at a time
#   ./scripts/start_overnight.sh --no-tests --experiment rq1     # skip the unit tests before starting
#   DRY_RUN=1 ./scripts/start_overnight.sh --experiment rq1      # print the plan and its time estimate only
#   ./scripts/start_overnight.sh --list                          # the experiments: what each asks, and its settings
#
# Background instead of a terminal (survives closing it; progress reports every 10 minutes):
#   setsid nohup env DEVICE=cuda ./scripts/start_overnight.sh --experiment rq1 > outputs/overnight_console.log 2>&1 &
#   tail -f outputs/overnight_console.log          # watch;  pkill -INT -f overnight_runs.py  # stop (saves)
#
# Checks that the virtual environment exists and the unit tests pass, warns about uncommitted changes
# (the runs use the files as they are on disk), records branch, commit, arguments and settings in
# <OUT>/run_info.txt, then runs scripts/overnight_runs.py in the foreground: it prints each run's purpose
# and progress, while the training runs as child processes logging to <OUT>/logs/. The machine is
# kept from sleeping while it runs. Closing the terminal stops the run, so for an overnight run use tmux
# (tmux new -s overnight; detach with Ctrl+B then D; reattach with tmux attach -t overnight).
# Ctrl+C stops the current run after it saves; running the same command again continues where it stopped.
# Arguments other than --no-tests (--experiment NAME, --list) and the settings of overnight_runs.py (environment
# variables: OUT, JOBS, DEVICE, SEEDS, OPERATOR_EMBEDDINGS, ...; see its header) pass through, e.g.
#   SEEDS="0 1" ./scripts/start_overnight.sh --experiment rq3-repair-minimal

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export OUT="${OUT:-outputs/overnight_$(date +%Y%m%d)}"
# Use only the virtual environment's packages: a PYTHONPATH from the shell (e.g. a sourced ROS setup) adds
# outside packages, including pytest plugins that fail to import in .venv (yaml, lark). The project does
# not need PYTHONPATH (pip install -e . puts src/ on the path).
unset PYTHONPATH
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1  # the tests use no pytest plugins
PYTHON=".venv/bin/python"
[[ -x "${PYTHON}" ]] || { echo "no virtual environment at .venv (see README: python -m venv .venv && pip install -e .)"; exit 1; }

run_tests=1
arguments=()
for argument in "$@"; do
  if [[ "${argument}" == "--no-tests" ]]; then run_tests=0; else arguments+=("${argument}"); fi
done
case " ${arguments[*]} " in  # describing the experiments needs no checks and records nothing
  *" --list "* | *" --help "* | *" -h "*) exec "${PYTHON}" scripts/overnight_runs.py "${arguments[@]}" ;;
esac

branch="$(git rev-parse --abbrev-ref HEAD)"
changes="$(git status --short)"
echo "branch ${branch} at $(git rev-parse --short HEAD)"
if [[ -n "${changes}" ]]; then
  echo "warning: uncommitted changes; the runs use the files as they are now:"
  echo "${changes}" | sed 's/^/  /'
fi
[[ -z "${TMUX:-}${STY:-}" && "${DRY_RUN:-0}" != "1" ]] && \
  echo "note: not inside tmux or screen; closing this terminal stops the runs (see the header of this script)"

if [[ "${run_tests}" == 1 && "${DRY_RUN:-0}" != "1" ]]; then
  echo "running unit tests ..."
  test_output="$("${PYTHON}" -m pytest -q -m unit -p no:cacheprovider -rfE 2>&1)" || {
    echo "${test_output}" | grep -E "^(FAILED|ERROR)|Error|error:" | head -20
    echo "${test_output}" | tail -1
    echo "unit tests failed; not starting. Full output: ${PYTHON} -m pytest -m unit"
    exit 1
  }
  echo "unit tests passed"
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  mkdir -p "${OUT}"
  {
    echo "started   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "branch    ${branch}"
    echo "commit    $(git rev-parse HEAD)"
    echo "changes   ${changes:-none}"
    echo "arguments ${arguments[*]:-none (the default experiment)}"
    echo "settings  $(env | grep -E '^(OUT|RUN_GROUPS|ALGORITHM|SKILL_SAMPLING|SEEDS|OPERATOR_EMBEDDINGS|REPAIR_HIDDEN_EMBEDDINGS|TEXT_BACKEND|BASE_SKILLS|REPAIRS|REPAIR_MODES|REPAIR_EVAL_INTERVAL|LEARN_STEPS|RQ1_STEPS|RQ1_BUDGET_MARGIN|REPAIR_STEPS|TRAIN_EVAL_INTERVAL|EVAL_EPISODES|SAVE_INTERVAL|DEVICE|JOBS|THREADS)=' | tr '\n' ' ')"
  } > "${OUT}/run_info.txt"
fi

# systemd-inhibit: no suspend while the runs go (if available); the runner itself stays in the foreground.
inhibit=()
command -v systemd-inhibit > /dev/null && inhibit=(systemd-inhibit --what=sleep:idle --who=overnight_runs --why="overnight experiments")
exec "${inhibit[@]}" "${PYTHON}" scripts/overnight_runs.py "${arguments[@]}"
