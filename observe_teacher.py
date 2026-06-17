"""Render and print the hardcoded/scripted teacher for each symbolic skill."""

from __future__ import annotations

import argparse
import time

import numpy as np

from fetch_blockworld.envs import make_skill_env
from fetch_blockworld.scripted import scripted_teacher_action
from fetch_blockworld.skills import SKILLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", choices=sorted(SKILLS), default="pickup")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.04)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = None if args.no_render else "human"
    env = make_skill_env(args.skill, render_mode=render_mode, seed=args.seed)

    try:
        for ep in range(args.episodes):
            # By default seed=None, so every reset restores the arm and samples a
            # new block pose from the env's RNG. If --seed is provided, episodes
            # are deterministic but still use different seeds.
            reset_seed = None if args.seed is None else args.seed + ep
            obs, info = env.reset(seed=reset_seed)
            print(f"\n=== teacher rollout skill={args.skill} episode={ep} seed={reset_seed} ===")
            print("initial numeric_state:", info.get("numeric_state", {}))
            print("initial facts:", _compact_facts(info.get("facts", {})))
            if "scripted_pickup_success" in info:
                print(
                    "scripted pickup init:",
                    {
                        "success": info["scripted_pickup_success"],
                        "steps": info["scripted_pickup_steps"],
                    },
                )

            for t in range(args.steps):
                action = scripted_teacher_action(args.skill, obs, env.evaluator)
                obs, reward, terminated, truncated, info = env.step(action)

                if t % args.print_every == 0 or terminated or truncated:
                    facts = info.get("facts", {})
                    print(
                        f"t={t:03d} "
                        f"action={np.round(action, 2)} "
                        f"reward={reward:.3f} "
                        f"success={bool(info.get('is_success', 0.0))} "
                        f"facts={_compact_facts(facts)}"
                    )

                if not args.no_render:
                    env.render()
                    time.sleep(args.sleep)

                if terminated or truncated:
                    break
    finally:
        env.close()


def _compact_facts(facts: dict[str, bool]) -> dict[str, bool]:
    """Print only true facts plus the important gripper/table negatives."""
    keep_when_false = {"object_on_table", "object_lifted", "holding_object", "gripper_open", "gripper_closed"}
    return {k: bool(v) for k, v in facts.items() if v or k in keep_when_false}


if __name__ == "__main__":
    main()
