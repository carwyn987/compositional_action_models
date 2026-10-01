#!/usr/bin/env python3
"""Render a Gymnasium-Robotics Fetch environment driven by a simple policy.

    python scripts/visualize_environment.py --num-blocks 5
    python scripts/visualize_environment.py --policy random
    python scripts/visualize_environment.py --env-id FetchPush-v4
    python scripts/visualize_environment.py --no-render --episodes 5

The scripted policy picks up block0.
Actions are [dx, dy, dz, gripper] in [-1, 1]; gripper > 0 opens, < 0 closes.
"""

import argparse
import time

import gymnasium as gym
import numpy as np

from cam.environments.fetch import fetch_multiblock_environment
from cam.environments.fetch.fetch_env_state_annotation_wrapper import FetchEnvStateAnnotationWrapper
from cam.environments.fetch.fetch_predicate_evaluation_wrapper import FetchPredicateEvaluationWrapper
from cam.environments.fetch.fetch_scripted_policies import FetchScriptedPickupPolicy
from cam.logging_config import configure_logging
from cam.policies.policy import RandomPolicy
from cam.skills.registry import build_skill

POLICIES = {
    "random": lambda env: RandomPolicy(env.action_space),
    "scripted-pickup": lambda env: FetchScriptedPickupPolicy(),
}
PICKUP_BLOCK0 = build_skill("pickup", {"symbolic_action_model_format": "pddl"}).ground({"?o": "block0"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-id", default=fetch_multiblock_environment.ENVIRONMENT_ID)
    parser.add_argument(
        "--num-blocks",
        type=int,
        help=f"blocks in {fetch_multiblock_environment.ENVIRONMENT_ID} (default 2)",
    )
    parser.add_argument("--policy", choices=sorted(POLICIES), default="scripted-pickup")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.02, help="seconds between rendered frames")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=960)
    args = parser.parse_args()
    if args.num_blocks is not None and args.env_id != fetch_multiblock_environment.ENVIRONMENT_ID:
        parser.error(f"--num-blocks applies only to --env-id {fetch_multiblock_environment.ENVIRONMENT_ID}")
    return args


def format_state(state) -> str:
    blocks = " ".join(f"{name}={np.round(position, 3)}" for name, position in state.block_positions.items())
    return f"gripper={np.round(state.gripper_position, 3)} fingers={state.finger_width:.3f} {blocks}"


def main() -> None:
    args = parse_args()
    configure_logging()
    environment_kwargs = {}
    if args.num_blocks is not None:
        environment_kwargs["num_blocks"] = args.num_blocks
    base_env = gym.make(
        args.env_id,
        render_mode=None if args.no_render else "human",
        width=args.window_width,
        height=args.window_height,
        max_episode_steps=args.max_episode_steps,
        **environment_kwargs,
    )
    env = FetchPredicateEvaluationWrapper(FetchEnvStateAnnotationWrapper(base_env))
    env.action_space.seed(args.seed)
    policy = POLICIES[args.policy](env)

    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            policy.reset()
            block_start_z = {name: p[2] for name, p in info["environment_state"].block_positions.items()}
            print(f"=== episode {episode} env={args.env_id} policy={args.policy} ===")

            for t in range(args.max_episode_steps):
                action = policy(obs, info, PICKUP_BLOCK0)
                obs, reward, terminated, truncated, info = env.step(action)
                if t % args.print_every == 0 or terminated or truncated:
                    print(f"t={t:03d} action={np.round(action, 2)} {format_state(info['environment_state'])}")
                if not args.no_render:
                    time.sleep(args.sleep)
                if terminated or truncated:
                    break

            for name, position in info["environment_state"].block_positions.items():
                print(f"{name} lifted {position[2] - block_start_z[name]:.3f} m")
    finally:
        env.close()


if __name__ == "__main__":
    main()
