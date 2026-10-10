#!/usr/bin/env bash
# Start the overnight experiments (scripts/overnight_runs.sh) in the background, from the current branch.
#
#   ./scripts/start_overnight.sh              # check, then start; returns immediately
#   ./scripts/start_overnight.sh --no-tests   # skip the unit tests before starting
#
# Before starting it checks that the virtual environment exists and the unit tests pass, and warns about
# uncommitted changes (the run uses the files as they are on disk). The run is detached from the terminal
# (closing the terminal or VS Code does not stop it) and keeps the machine from sleeping while it runs.
# Settings of overnight_runs.sh (OUT, RUN_GROUPS, CONDITIONS, SEEDS, *_STEPS, ...) pass through, e.g.
#   SEEDS="0 1" ./scripts/start_overnight.sh
#
# Writes to OUT (default outputs/overnight_<date>):
#   run_info.txt    branch, commit, uncommitted changes and settings the run started from
#   overnight.log   the runner's output (progress, then the summary)
#   overnight.pid   its process group id, used to detect a run already in progress and to stop it
# Follow it with:   tail -f <OUT>/overnight.log
# Stop it with:     kill -- -$(cat <OUT>/overnight.pid)
# Results:          <OUT>/review.txt (written at the end), or ./scripts/review_runs.py <OUT> at any time

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

pid_file="${OUT}/overnight.pid"
if [[ -f "${pid_file}" ]] && kill -0 -- "-$(cat "${pid_file}")" 2>/dev/null; then
  echo "a run is already in progress in ${OUT} (process group $(cat "${pid_file}")); stop it with: kill -- -$(cat "${pid_file}")"
  exit 1
fi

branch="$(git rev-parse --abbrev-ref HEAD)"
commit="$(git rev-parse --short HEAD)"
changes="$(git status --short)"
echo "branch ${branch} at ${commit}"
if [[ -n "${changes}" ]]; then
  echo "warning: uncommitted changes; the run uses the files as they are now:"
  echo "${changes}" | sed 's/^/  /'
fi

if [[ "${1:-}" != "--no-tests" ]]; then
  echo "running unit tests ..."
  test_output="$("${PYTHON}" -m pytest -q -m unit -p no:cacheprovider -rfE 2>&1)" || {
    echo "${test_output}" | grep -E "^(FAILED|ERROR)|Error|error:" | head -20
    echo "${test_output}" | tail -1
    echo "unit tests failed; not starting. Full output: ${PYTHON} -m pytest -m unit"
    exit 1
  }
  echo "unit tests passed"
fi

mkdir -p "${OUT}"
{
  echo "started   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "branch    ${branch}"
  echo "commit    $(git rev-parse HEAD)"
  echo "changes   ${changes:-none}"
  echo "settings  $(env | grep -E '^(OUT|JOBS|RUN_GROUPS|ALGORITHM|SEEDS|CONDITIONS|ALIAS_CONDITIONS|BASE_SKILLS|REPAIR_FROM|REPAIR_TO|LEARN_STEPS|RQ1_STEPS|REPAIR_STEPS|EVAL_INTERVAL|EVAL_EPISODES|SAVE_INTERVAL)=' | tr '\n' ' ')"
} > "${OUT}/run_info.txt"

# setsid: own process group, so it survives the terminal and can be stopped as a group;
# systemd-inhibit: no suspend while it runs (if available).
inhibit=()
command -v systemd-inhibit > /dev/null && inhibit=(systemd-inhibit --what=sleep:idle --who=overnight_runs --why="overnight experiments")
setsid nohup "${inhibit[@]}" ./scripts/overnight_runs.sh > "${OUT}/overnight.log" 2>&1 < /dev/null &
echo $! > "${pid_file}"

sleep 2
if ! kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
  echo "the run exited immediately; see ${OUT}/overnight.log:"
  tail -5 "${OUT}/overnight.log"
  exit 1
fi
head -4 "${OUT}/overnight.log"
echo
echo "started in the background (process group $(cat "${pid_file}")); output in ${OUT}"
echo "  follow:   tail -f ${OUT}/overnight.log"
echo "  stop:     kill -- -$(cat "${pid_file}")"
echo "  results:  ${OUT}/review.txt when done, or ./scripts/review_runs.py ${OUT} at any time"
