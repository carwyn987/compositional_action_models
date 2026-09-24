"""Experimental test domain for the multi-block Fetch environment.

Verifies that we can spawn and manipulate multiple identified blocks:
  * both env ids build and have the expected observation dimensionality,
  * the stack env spawns blocks separated on the table,
  * the unstack env spawns the mover already stacked on the base,
  * the mover/base identifiers are present and valid,
  * the scripted teachers can actually stack / unstack the correct blocks.

Usage:
  python experimental/test_multiblock.py
  python experimental/test_multiblock.py --episodes 5
  MUJOCO_GL=glfw python experimental/test_multiblock.py --render
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Allow running as `python experimental/test_multiblock.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fetch_blockworld.envs import make_skill_env
from fetch_blockworld.scripted import scripted_teacher_action


def _block_positions(info: dict) -> list[list[float]]:
    return info["numeric_state"]["block_positions"]


def run_skill(skill: str, episodes: int, render: bool, max_steps: int = 200) -> bool:
    env = make_skill_env(skill, render_mode="human" if render else None, seed=0)
    num_blocks = env.unwrapped.num_blocks
    expected_dim = 25 + 9 * (num_blocks - 1) + 2 * num_blocks

    obs, info = env.reset(seed=0)
    obs_dim = obs["observation"].shape[0]
    assert obs_dim == expected_dim, f"{skill}: obs dim {obs_dim} != expected {expected_dim}"

    mover = info["numeric_state"]["mover_index"]
    base = info["numeric_state"]["base_index"]
    assert mover is not None and base is not None and mover != base, (
        f"{skill}: invalid mover/base ids {mover}/{base}"
    )

    positions = _block_positions(info)
    f0 = info["facts"]
    print(f"\n=== {skill} ===")
    print(f"  num_blocks={num_blocks} obs_dim={obs_dim} (expected {expected_dim})")
    print(f"  mover_index={mover} base_index={base}")
    for i, p in enumerate(positions):
        tag = " (mover)" if i == mover else " (base)" if i == base else ""
        print(f"  object{i} pos={np.round(p, 3).tolist()}{tag}")

    if skill == "stack":
        assert not f0["blocks_stacked"], "stack should start unstacked"
        assert f0["mover_clear_of_base"], "stack blocks should start separated"
    else:  # unstack
        assert f0["blocks_stacked"], "unstack should start stacked"
        assert not f0["mover_clear_of_base"], "unstack mover should start over the base"

    successes = 0
    for ep in range(episodes):
        obs, info = env.reset(seed=1000 + ep)
        done = False
        success = False
        steps = 0
        while not done and steps < max_steps:
            action = scripted_teacher_action(skill, obs, env.evaluator)
            obs, reward, terminated, truncated, info = env.step(action)
            success = bool(info["is_success"])
            done = terminated or truncated
            steps += 1
        successes += int(success)
        print(f"  episode {ep}: success={success} steps={steps}")

    env.close()
    ok = successes == episodes
    print(f"  scripted teacher: {successes}/{episodes} success -> {'OK' if ok else 'FAIL'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    results = {
        "stack": run_skill("stack", args.episodes, args.render),
        "unstack": run_skill("unstack", args.episodes, args.render),
    }

    print("\n==== SUMMARY ====")
    for skill, ok in results.items():
        print(f"  {skill}: {'PASS' if ok else 'FAIL'}")
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
