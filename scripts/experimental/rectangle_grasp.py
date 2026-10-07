#!/usr/bin/env python3
"""EXPERIMENTAL (temporary): Fetch picking up a rectangular block, which needs gripper yaw.

The block is 0.12 x 0.04 x 0.05 m. Its long side is wider than the gripper can
open (about 0.10 m), so it can only be grasped across its short side: the
gripper must rotate about the vertical axis to match the block's random yaw.

Stock Fetch cannot do this: its action is [dx, dy, dz, gripper] and its wrist
orientation is pinned to a fixed quaternion every step. FetchRectangleEnv adds a
yaw command, action [dx, dy, dz, dyaw, gripper], by rotating the orientation
target of the mocap weld that drives the gripper; the arm's wrist roll joint
follows. The gripper's yaw is appended to obs["observation"].

    ./scripts/experimental/rectangle_grasp.py --episodes 10              # yaw-aware scripted pickup
    ./scripts/experimental/rectangle_grasp.py --episodes 10 --no-rotate  # same, gripper yaw fixed at 0
    ./scripts/experimental/rectangle_grasp.py --min-block-yaw 70 --no-rotate   # orientations that need rotation
    LIBGL_ALWAYS_SOFTWARE=1 ./scripts/experimental/rectangle_grasp.py --render --episodes 3
"""

import argparse
import os
import tempfile
import time

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium_robotics.envs.fetch import MujocoFetchEnv

from cam.environments.fetch.fetch_multiblock_environment import BLOCK_HALF_SIZE, build_multiblock_xml

HALF_EXTENTS = (0.06, 0.02, 0.025)  # rectangle half sizes: long x, short y, height z
FIXED_GRIPPER_QUAT = np.array([1.0, 0.0, 1.0, 0.0]) / np.sqrt(2)  # Fetch's pinned wrist orientation
MAX_YAW_STEP = 0.1  # radians of gripper yaw per step at |dyaw| = 1
TABLE_REST_Z = 0.425


def yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class FetchRectangleEnv(MujocoFetchEnv):
    """Fetch pick-and-place with one rectangular block at a random yaw, and gripper yaw control.

    The block's yaw is sampled with |yaw| in [min_block_yaw, pi/2]. Closing fingers turn a
    free block into alignment unless it is far from the finger axis (more than about 66
    degrees for these dimensions), so a large min_block_yaw makes rotating the gripper necessary.
    """

    def __init__(self, min_block_yaw: float = 0.0, **kwargs):
        self.gripper_yaw = 0.0
        self.min_block_yaw = min_block_yaw
        cube = f'size="{BLOCK_HALF_SIZE} {BLOCK_HALF_SIZE} {BLOCK_HALF_SIZE}"'
        xml = build_multiblock_xml(1).replace(cube, 'size="{} {} {}"'.format(*HALF_EXTENTS))
        with tempfile.NamedTemporaryFile("w", prefix="fetch_rectangle_", suffix=".xml", delete=False) as f:
            f.write(xml)
        try:
            super().__init__(
                model_path=f.name,
                has_object=True,
                block_gripper=False,
                n_substeps=20,
                gripper_extra_height=0.2,
                target_in_the_air=True,
                target_offset=0.0,
                obj_range=0.15,
                target_range=0.15,
                distance_threshold=0.05,
                initial_qpos={
                    "robot0:slide0": 0.405,
                    "robot0:slide1": 0.48,
                    "robot0:slide2": 0.0,
                    "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
                },
                reward_type="sparse",
                **kwargs,
            )
        finally:
            os.remove(f.name)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (5,), np.float32)

    def _set_action(self, action):
        dx, dy, dz, dyaw, gripper = np.asarray(action, dtype=np.float64)
        super()._set_action(np.array([dx, dy, dz, gripper]))  # position and fingers as in stock Fetch
        self.gripper_yaw = float(np.clip(self.gripper_yaw + MAX_YAW_STEP * dyaw, -np.pi / 2, np.pi / 2))
        target = np.zeros(4)
        mujoco.mju_mulQuat(target, yaw_quaternion(self.gripper_yaw), FIXED_GRIPPER_QUAT)
        self.data.mocap_quat[0] = target  # replaces the pinned orientation; the weld turns the wrist

    def _reset_sim(self):
        self._mujoco.mj_resetData(self.model, self.data)
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None
        self.gripper_yaw = 0.0

        gripper_xy = self.initial_gripper_xpos[:2]
        while True:
            xy = gripper_xy + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            if np.linalg.norm(xy - gripper_xy) >= 0.1:
                break
        block_yaw = self.np_random.choice([-1, 1]) * self.np_random.uniform(self.min_block_yaw, np.pi / 2)
        qpos = self._utils.get_joint_qpos(self.model, self.data, "object0:joint")
        qpos[:3] = [xy[0], xy[1], self.height_offset]
        qpos[3:] = yaw_quaternion(block_yaw)
        self._utils.set_joint_qpos(self.model, self.data, "object0:joint", qpos)
        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _get_obs(self):
        obs = super()._get_obs()
        obs["observation"] = np.concatenate([obs["observation"], [self.gripper_yaw]])
        return obs


def short_side_yaw(block_yaw: float) -> float:
    """Gripper yaw whose finger axis lies along the block's short side, in [-pi/2, pi/2).

    At gripper yaw 0 the fingers close along world y; the block's short side is
    its local y, rotated by its yaw. The rectangle is symmetric under a half turn.
    """
    return (block_yaw + np.pi / 2) % np.pi - np.pi / 2


class ScriptedRectanglePickup:
    """Rotate to the block's short side while moving above it, descend, close, lift."""

    APPROACH_HEIGHT, LIFT_HEIGHT, TOLERANCE, YAW_TOLERANCE, CLOSE_STEPS, GAIN = 0.10, 0.15, 0.01, 0.03, 10, 10.0

    def __init__(self, rotate: bool):
        self.rotate = rotate

    def reset(self):
        self.phase, self.close_steps, self.block_start, self.target_yaw = "approach", 0, None, None

    def __call__(self, obs):
        raw = obs["observation"]
        gripper, block, block_yaw, gripper_yaw = raw[0:3], raw[3:6], raw[13], raw[-1]
        if self.block_start is None:
            # Fixed at the start: a held block lags the gripper slightly, so steering toward
            # its current yaw while carrying it would rotate both without end.
            self.block_start = block.copy()
            self.target_yaw = short_side_yaw(block_yaw) if self.rotate else 0.0
        target_yaw = self.target_yaw
        dyaw = float(np.clip((target_yaw - gripper_yaw) / MAX_YAW_STEP, -1, 1))

        def servo(target, gripper_command):
            return np.array([*np.clip(self.GAIN * (target - gripper), -1, 1), dyaw, gripper_command], np.float32)

        if self.phase == "approach":
            target = block + [0, 0, self.APPROACH_HEIGHT]
            if np.linalg.norm(gripper - target) < self.TOLERANCE and abs(target_yaw - gripper_yaw) < self.YAW_TOLERANCE:
                self.phase = "descend"
            return servo(target, 1.0)
        if self.phase == "descend":
            if np.linalg.norm(gripper - block) < self.TOLERANCE:
                self.phase = "close"
            return servo(block, 1.0)
        if self.phase == "close":
            self.close_steps += 1
            if self.close_steps >= self.CLOSE_STEPS:
                self.phase = "lift"
            return np.array([0, 0, 0, dyaw, -1.0], np.float32)
        return servo(self.block_start + [0, 0, self.LIFT_HEIGHT], -1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-rotate", action="store_true", help="keep gripper yaw at 0 (stock Fetch behaviour)")
    parser.add_argument(
        "--min-block-yaw", type=float, default=0.0,
        help="degrees; sample block yaw with |yaw| >= this (e.g. 70: grasping needs gripper rotation)",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.02)
    args = parser.parse_args()

    env = gym.wrappers.TimeLimit(
        FetchRectangleEnv(
            min_block_yaw=np.radians(args.min_block_yaw),
            render_mode="human" if args.render else None,
            width=1280,
            height=960,
        ),
        args.max_steps,
    )
    policy = ScriptedRectanglePickup(rotate=not args.no_rotate)
    successes = 0
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            policy.reset()
            block_yaw = obs["observation"][13]
            for _ in range(args.max_steps):
                obs, _, terminated, truncated, _ = env.step(policy(obs))
                if args.render:
                    time.sleep(args.sleep)
                if terminated or truncated:
                    break
            raw = obs["observation"]
            lift = raw[5] - TABLE_REST_Z
            held = lift > 0.05 and np.linalg.norm(raw[0:3] - raw[3:6]) < 0.03
            successes += held
            print(
                f"episode {episode}: block yaw {np.degrees(block_yaw):6.1f} deg, gripper yaw "
                f"{np.degrees(raw[-1]):6.1f} deg, lift {lift:.3f} m -> {'picked up' if held else 'failed'}"
            )
    finally:
        env.close()
    print(f"success: {successes}/{args.episodes} ({'fixed yaw' if args.no_rotate else 'yaw-aware'})")


if __name__ == "__main__":
    main()
