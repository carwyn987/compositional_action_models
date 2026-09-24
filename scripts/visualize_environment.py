"""Render a Gymnasium-Robotics Fetch environment driven by a simple policy.

    python scripts/visualize_environment.py --policy scripted-pickup
    python scripts/visualize_environment.py --policy random --env-id FetchPush-v4
    python scripts/visualize_environment.py --no-render --episodes 5

Fetch observation["observation"] layout used by the scripted policy:
    0:3  gripper position, 3:6 object position, 9:11 finger joint positions.
Actions are [dx, dy, dz, gripper] in [-1, 1]; gripper > 0 opens, < 0 closes.
"""

import argparse
import time

import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

OPEN, CLOSE = 1.0, -1.0


class RandomPolicy:
    def __init__(self, action_space: gym.Space):
        self.action_space = action_space

    def reset(self) -> None:
        pass

    def __call__(self, obs: dict) -> np.ndarray:
        return self.action_space.sample()


class ScriptedPickupPolicy:
    """Approach above the block, descend, close, lift. Deterministic given obs."""

    APPROACH_HEIGHT = 0.10
    LIFT_HEIGHT = 0.15
    REACHED_TOLERANCE = 0.01
    CLOSE_STEPS = 10
    GAIN = 10.0

    def reset(self) -> None:
        self.phase = "approach"
        self.close_steps_taken = 0
        self.block_start = None

    def __call__(self, obs: dict) -> np.ndarray:
        gripper = obs["observation"][0:3]
        block = obs["observation"][3:6]
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
        return float(np.linalg.norm(gripper - target)) < self.REACHED_TOLERANCE

    def _servo(self, gripper: np.ndarray, target: np.ndarray, gripper_command: float) -> np.ndarray:
        delta = np.clip(self.GAIN * (target - gripper), -1.0, 1.0)
        return np.array([*delta, gripper_command], dtype=np.float32)


POLICIES = {
    "random": lambda env: RandomPolicy(env.action_space),
    "scripted-pickup": lambda env: ScriptedPickupPolicy(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-id", default="FetchPickAndPlace-v4")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="scripted-pickup")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.02, help="seconds between rendered frames")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = gym.make(
        args.env_id,
        render_mode=None if args.no_render else "human",
        max_episode_steps=args.max_episode_steps,
    )
    env.action_space.seed(args.seed)
    policy = POLICIES[args.policy](env)

    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            policy.reset()
            block_start_z = float(obs["observation"][5])
            print(f"=== episode {episode} env={args.env_id} policy={args.policy} ===")

            for t in range(args.max_episode_steps):
                action = policy(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                if t % args.print_every == 0 or terminated or truncated:
                    gripper = np.round(obs["observation"][0:3], 3)
                    block = np.round(obs["observation"][3:6], 3)
                    print(f"t={t:03d} action={np.round(action, 2)} gripper={gripper} block={block} reward={reward:.2f}")
                if not args.no_render:
                    time.sleep(args.sleep)
                if terminated or truncated:
                    break

            lift = float(obs["observation"][5]) - block_start_z
            print(f"block lifted {lift:.3f} m")
    finally:
        env.close()


if __name__ == "__main__":
    main()
