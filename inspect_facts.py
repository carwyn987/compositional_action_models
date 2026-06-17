"""Smoke-test the environment and print symbolic facts under random actions."""

from __future__ import annotations

import argparse

from fetch_blockworld.envs import make_skill_env
from fetch_blockworld.skills import SKILLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", choices=sorted(SKILLS), default="pickup")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = make_skill_env(args.skill, render_mode="human" if args.render else None, seed=0)
    obs, info = env.reset(seed=0)
    print("initial facts:", info["facts"])
    print("initial numeric_state:", info["numeric_state"])

    for t in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if t % 10 == 0 or terminated or truncated:
            print(f"t={t:03d} reward={reward:.3f} success={info['is_success']} facts={info['facts']}")
        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
