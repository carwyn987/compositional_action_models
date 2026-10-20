#!/usr/bin/env python3
"""Is a trained policy stuck, or slowly getting to its goal? Runs it past the training step limit and tracks how far
each episode is from the skill's goal.

    ./scripts/check_policy_progress.py outputs/overnight_<date>/learnability/*/*/     # every learnability run
    ./scripts/check_policy_progress.py outputs/<run> --skills pickup stack --episodes 20 --max-steps 300

Each run's skills (default: all it trained) are played deterministically for --max-steps (default 300, three times
the training limit) on the run's evaluation scenes. The distance to the goal is:
    pickup, unstack, raised variants   gripper to block, plus the lift still missing
    putdown, stack, place-beside       block to where it should rest, plus how far the fingers are from open
    pushes                             block to its push goal, plus gripper to block
Each episode gets a verdict:
    success at step N        after step 100: slower than training allowed, but it gets there
    failed at step N         with the failure
    progressing              the distance fell by 5 mm or more in the last 30 steps
    stuck since step N       no 5 mm improvement after step N; the gripper frozen, or moving without getting closer
with the distance (cm) at the start, at step 100 and at the end, the finger opening at the end and whether the
closed fingers are around the block (gripped), and the policy's action noise (the mean standard deviation of its
Gaussian, before tanh) at the first and last step: how much it still explores where it ends up.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from cam.environments.fetch.fetch_multiblock_environment import BLOCK_HALF_SIZE
from cam.environments.fetch.fetch_predicate_evaluation_wrapper import (
    BESIDE_SPACING,
    OPEN_FINGER_WIDTH,
    RAISED_HEIGHT,
    TABLE_REST_Z,
    gripping,
    push_goal,
)
from cam.experiments.environment_setup import setup_environment
from cam.policies.stable_baselines3_policy import ALGORITHMS
from cam.skills.registry import build_skill

IMPROVEMENT = 0.005  # metres the distance must fall by to count as progress
RECENT = 30  # final steps that decide progressing vs stuck
EVALUATION_SEED_OFFSET = 10_000  # main.py evaluates on scenes seeded with the run's seed + this


def goal_distance(skill: str, state, grounded, start: np.ndarray) -> float:
    """How far the scene is from the grounded action's goal (metres); start is the moved block's starting position."""
    gripper, block = state.gripper_position, state.block_positions[grounded.arguments[0]]
    to_open = max(0.0, OPEN_FINGER_WIDTH - float(state.finger_width))
    if skill in ("pickup", "unstack"):  # holding needs the block more than 1 cm off its support
        return float(np.linalg.norm(gripper - block) + max(0.0, start[2] + 0.02 - block[2]))
    if skill in ("pickup-raised", "unstack-raised"):
        return float(np.linalg.norm(gripper - block) + abs(TABLE_REST_Z + np.mean(RAISED_HEIGHT) - block[2]))
    if skill == "putdown":
        return float(max(0.0, block[2] - TABLE_REST_Z) + to_open)
    if skill == "stack":
        base = state.block_positions[grounded.arguments[1]]
        return float(np.linalg.norm(block - (base + [0.0, 0.0, 2 * BLOCK_HALF_SIZE])) + to_open)
    if skill == "place-beside":
        base = state.block_positions[grounded.arguments[1]]
        return float(abs(np.linalg.norm(block[:2] - base[:2]) - BESIDE_SPACING) + max(0.0, block[2] - TABLE_REST_Z)
                     + to_open)
    if skill.startswith("push"):
        return float(np.linalg.norm(block[:2] - push_goal(state, grounded)) + np.linalg.norm(gripper - block))
    return float(np.linalg.norm(gripper - block))


def run_episode(env, skill: str, act, max_steps: int, seed: int | None = None, noise=None) -> tuple[str, str]:
    """One episode of skill: (verdict: success / failed / progressing / stuck, report line). act(obs, info) -> action;
    noise(obs) -> the policy's action standard deviation, or None to leave it out."""
    obs, info = env.reset(seed=seed, options={"skill": skill})
    grounded = info["grounded_action_model"]
    start = np.asarray(info["environment_state"].block_positions[grounded.arguments[0]], dtype=np.float64)
    noise_start = noise(obs) if noise else None
    distances, grippers, best, last_progress = [], [], np.inf, 0
    for step in range(1, max_steps + 1):
        obs, _, terminated, truncated, info = env.step(act(obs, info))
        state = info["environment_state"]
        distances.append(goal_distance(skill, state, grounded, start))
        grippers.append(np.asarray(state.gripper_position, dtype=np.float64))
        if distances[-1] < best - IMPROVEMENT:
            best, last_progress = distances[-1], step
        if terminated or truncated:
            break
    moved = float(np.linalg.norm(grippers[-1] - grippers[max(0, len(grippers) - 1 - RECENT)]))
    if info["is_success"]:
        kind, verdict = "success", f"success at step {step}" + (" (after the training limit)" if step > 100 else "")
    elif info.get("failure"):
        kind, verdict = "failed", f"failed at step {step}: {info['failure']}"
    elif step - last_progress <= RECENT:
        kind, verdict = "progressing", f"progressing (closer by 5 mm+ in the last {RECENT} steps)"
    else:
        kind = "stuck"
        verdict = (f"stuck since step {last_progress}: gripper "
                   + ("frozen" if moved < 0.01 else f"moving without getting closer ({100 * moved:.0f} cm)")
                   + f" in the last {RECENT} steps")
    at_100 = f"{100 * distances[99]:.1f}" if len(distances) >= 100 else "-"
    fingers = f"fingers {100 * float(state.finger_width):.1f} cm" + (
        ", gripped" if gripping(state, grounded.arguments[0]) else "")
    line = (f"{verdict:72s} distance {100 * distances[0]:5.1f} → {at_100:>5s} → {100 * distances[-1]:5.1f} cm"
            f"   {fingers:22s}")
    if noise_start is not None:
        line += f"   noise {noise_start:.2f} → {noise(obs):.2f}"
    return kind, line


def check_run(run: Path, skills: list[str] | None, episodes: int, max_steps: int) -> None:
    config = json.loads((run / "config.json").read_text()) | {"render": False, "max_steps_per_episode": max_steps}
    skills = skills or config["skills"]
    model = ALGORITHMS[config["algorithm"]].load(run / "model.zip", device="cpu")
    env = setup_environment(config, [build_skill(name, config) for name in dict.fromkeys([*config["skills"], *skills])])

    def act(obs, info):
        return model.predict(obs, deterministic=True)[0]

    def noise(obs):
        with torch.no_grad():
            tensor, _ = model.policy.obs_to_tensor(obs)
            _, log_std, _ = model.actor.get_action_dist_params(tensor)
        return float(log_std.exp().mean())

    print(f"== {run}")
    try:
        for skill in skills:
            print(f"   -- {skill}")
            kinds = []
            for episode in range(episodes):
                seed = config["seed"] + EVALUATION_SEED_OFFSET if episode == 0 else None
                kind, line = run_episode(env, skill, act, max_steps, seed, noise if hasattr(model, "actor") else None)
                kinds.append(kind)
                print(f"      {line}", flush=True)
            print("      " + ", ".join(f"{kinds.count(kind)} {kind}" for kind in ("success", "progressing", "stuck",
                                                                                  "failed")))
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories (with config.json and model.zip)")
    parser.add_argument("--skills", nargs="+", help="skills to check (default: each run's skills)")
    parser.add_argument("--episodes", type=int, default=10, help="episodes per skill")
    parser.add_argument("--max-steps", type=int, default=300, help="steps per episode (training allows 100)")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    for run in args.runs:
        check_run(run, args.skills, args.episodes, args.max_steps)


if __name__ == "__main__":
    main()
