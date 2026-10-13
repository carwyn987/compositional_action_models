#!/usr/bin/env python3
"""Watch the best trained policy for each skill in a folder of runs (e.g. an overnight run, even while it runs).

    ./scripts/watch_best_policies.py                                   # newest outputs/overnight_*, every skill
    ./scripts/watch_best_policies.py --skills pickup stack unstack-raised --episodes 3
    ./scripts/watch_best_policies.py --folder outputs/overnight_20261011 --no-render   # success only, no window

For each skill, the best policy is the run whose latest evaluation of that skill has the highest deterministic
success rate (ties: the run trained longest). The selection is printed first, then each skill is played with
scripts/run_skill.py. Runs save every --save-interval steps, so a run still training shows its latest save.

Only one folder is searched (default: the newest outputs/overnight_*), because runs saved by older code may
not load with the current code (e.g. a smaller --operator-embedding-dim, or other predicate ids). The viewer
runs on the CPU (CUDA_VISIBLE_DEVICES unset), so it does not compete with training for the GPU. Ctrl+C skips
the current skill.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, help="folder of runs (default: the newest outputs/overnight_*)")
    parser.add_argument("--skills", nargs="+", help="only these skills (default: every skill found)")
    parser.add_argument("--episodes", type=int, default=2, help="episodes per skill")
    parser.add_argument("--no-render", action="store_true", help="no viewer window: print each episode's outcome only")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of the most likely one")
    parser.add_argument("--sleep", type=float, default=0.02, help="seconds between rendered frames")
    return parser.parse_args()


def best_runs(folder: Path, skills: list[str] | None) -> dict[str, tuple[float, int, Path]]:
    """skill -> (latest deterministic success rate, step of that evaluation, run directory)."""
    best: dict[str, tuple[float, int, Path]] = {}
    for metrics_path in folder.glob("**/metrics.json"):
        run = metrics_path.parent
        if not (run / "model.zip").exists():
            continue
        points = json.loads(metrics_path.read_text())["curves"]["deterministic"]["points"]
        for skill in {skill for point in points for skill in point["result"]}:
            if skills and skill not in skills:
                continue
            last = [point for point in points if skill in point["result"]][-1]
            candidate = (last["result"][skill]["success_rate"], last["step"], run)
            if skill not in best or candidate[:2] > best[skill][:2]:
                best[skill] = candidate
    return best


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    folder = args.folder or max((path for path in Path("outputs").glob("overnight_*") if path.is_dir()), default=None)
    if folder is None or not folder.is_dir():
        sys.exit("no folder of runs: pass --folder, or run scripts/start_overnight.sh first")
    best = best_runs(folder, args.skills)
    order = [skill for skill in (args.skills or sorted(best)) if skill in best]
    print(f"Best policy per skill in {folder}:")
    for skill in args.skills or sorted(best):
        if skill in best:
            rate, step, run = best[skill]
            print(f"  {skill:18s} success {rate:.2f} at step {step:>9,}  {run.relative_to(folder)}")
        else:
            print(f"  {skill:18s} no saved run yet")
    environment = os.environ | {"CUDA_VISIBLE_DEVICES": ""}
    for skill in order:
        rate, step, run = best[skill]
        print(f"\n== {skill}: success {rate:.2f} at step {step:,} ({run})", flush=True)
        command = [sys.executable, "scripts/run_skill.py", "--policy", "model", "--run-directory", str(run),
                   "--skill", skill, "--episodes", str(args.episodes), "--sleep", str(args.sleep)]
        command += ([] if args.no_render else ["--render"]) + (["--stochastic"] if args.stochastic else [])
        try:
            result = subprocess.run(command, env=environment)
        except KeyboardInterrupt:
            print(f"   skipped {skill}")
            continue
        if result.returncode != 0:
            print(f"   could not play {skill} from this run (exit code {result.returncode}); see the error above")


if __name__ == "__main__":
    main()
