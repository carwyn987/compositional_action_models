#!/usr/bin/env python3
"""Render a Gymnasium-Robotics Fetch environment driven by a simple policy.

    python scripts/visualize_environment.py --num-blocks 5
    python scripts/visualize_environment.py --policy random
    python scripts/visualize_environment.py --env-id FetchPush-v4
    python scripts/visualize_environment.py --no-render --episodes 5

Policies act on the FetchState in info["environment_state"].
Actions are [dx, dy, dz, gripper] in [-1, 1]; gripper > 0 opens, < 0 closes.
"""

import argparse
import time

import gymnasium as gym
import numpy as np

from cam.environments.fetch import fetch_multiblock_environment
from cam.environments.fetch.fetch_env_state_annotation_wrapper import FetchEnvStateAnnotationWrapper, FetchState
from cam.environments.fetch.fetch_predicate_evaluation_wrapper import FetchPredicateEvaluationWrapper

OPEN, CLOSE = 1.0, -1.0


class RandomPolicy:
    def __init__(self, action_space: gym.Space):
        self.action_space = action_space

    def reset(self) -> None:
        pass

    def __call__(self, state: FetchState) -> np.ndarray:
        return self.action_space.sample()


class ScriptedPickupPolicy:
    """Approach above the target block, descend, close, lift. Deterministic given states."""

    APPROACH_HEIGHT = 0.10
    LIFT_HEIGHT = 0.15
    REACHED_TOLERANCE = 0.01
    CLOSE_STEPS = 10
    GAIN = 10.0

    def __init__(self, block: str = "block0"):
        self.block = block

    def reset(self) -> None:
        self.phase = "approach"
        self.close_steps_taken = 0
        self.block_start = None

    def __call__(self, state: FetchState) -> np.ndarray:
        gripper = state.gripper_position
        block = state.block_positions[self.block]
        if self.block_start is None:
            self.block_start = block.copy()

        if self.phase == "approach":
            target = block + [0.0, 0.0, self.APPROACH_HEIGHT]
            if self._reached(gripper, target):
                self.phase = "descend"
            return self._servo(gripper, target, OPEN)
        if self.phase == "descend":
            if self._reached(gripper, block):
                self.phase = "close"
            return self._servo(gripper, block, OPEN)
        if self.phase == "close":
            self.close_steps_taken += 1
            if self.close_steps_taken >= self.CLOSE_STEPS:
                self.phase = "lift"
            return np.array([0.0, 0.0, 0.0, CLOSE], dtype=np.float32)
        return self._servo(gripper, self.block_start + [0.0, 0.0, self.LIFT_HEIGHT], CLOSE)

    def _reached(self, gripper: np.ndarray, target: np.ndarray) -> bool:
        """True when the gripper is within REACHED_TOLERANCE of target."""
        return float(np.linalg.norm(gripper - target)) < self.REACHED_TOLERANCE

    def _servo(self, gripper: np.ndarray, target: np.ndarray, gripper_command: float) -> np.ndarray:
        """Proportional step toward target (clipped to [-1, 1]), with the given gripper command."""
        delta = np.clip(self.GAIN * (target - gripper), -1.0, 1.0)
        return np.array([*delta, gripper_command], dtype=np.float32)


POLICIES = {
    "random": lambda env: RandomPolicy(env.action_space),
    "scripted-pickup": lambda env: ScriptedPickupPolicy(),
}


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


def format_facts(facts) -> str:
    return " ".join(sorted(f"({' '.join((fact.name, *fact.args))})" for fact in facts)) or "-"


def main() -> None:
    args = parse_args()
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
            print(f"facts at start: {format_facts(info['facts'])}")

            for t in range(args.max_episode_steps):
                action = policy(info["environment_state"])
                previous_facts = info["facts"]
                obs, reward, terminated, truncated, info = env.step(action)
                if info["facts"] != previous_facts:
                    print(
                        f"t={t:03d} facts added: {format_facts(info['facts'] - previous_facts)}"
                        f"  removed: {format_facts(previous_facts - info['facts'])}"
                    )
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
