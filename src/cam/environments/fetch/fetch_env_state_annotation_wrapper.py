"""Gymnasium-Robotics Fetch environments annotated with a named state."""

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np


@dataclass(frozen=True, eq=False)
class FetchState:
    gripper_position: np.ndarray  # (3,) x, y, z
    finger_width: float  # sum of the two finger joint positions
    block_positions: dict[str, np.ndarray]  # block name -> (3,) x, y, z
    block_rotations: dict[str, np.ndarray] = field(default_factory=dict)  # block name -> (3,) Euler angles


class FetchEnvStateAnnotationWrapper(gym.Wrapper):
    """Adds to info:
        environment_state   FetchState of the current observation
        objects             the scene's objects by name and type ({"block0": "block", ...})
        object_features     per object, its position and position relative to the gripper (6,)

    Works for the stock single-block Fetch object tasks and FetchMultiBlock-v0.
    Observations keep only obs["observation"]: Fetch's achieved_goal (a copy of
    block0's position) and desired_goal (a random target for Fetch's own task)
    are dropped. Layout of obs["observation"]:
        0:3    gripper position
        3:6    block0 position
        9:11   finger joint positions
        11:14  block0 rotation (Euler angles)
        25+9k  block k+1 position (3), relative position (3), rotation (3)
    Blocks are named block0..block{N-1}; N is determined by the observation length.
    """

    OBJECT_TYPES = ["block"]
    OBJECT_FEATURE_DIM = 6  # position (3) + position relative to the gripper (3)
    BASE_OBSERVATION_SIZE = 25
    EXTRA_BLOCK_SIZE = 9

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.num_blocks = self.count_blocks(env.observation_space["observation"].shape[0])
        self.observation_space = gym.spaces.Dict({"observation": env.observation_space["observation"]})

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return {"observation": obs["observation"]}, {**info, **self._annotations(obs)}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = {**info, **self._annotations(obs)}
        return {"observation": obs["observation"]}, reward, terminated, truncated, info

    def _annotations(self, obs: dict) -> dict:
        state = self.extract_state(obs)
        return {
            "environment_state": state,
            "objects": {name: "block" for name in state.block_positions},
            "object_features": {
                name: np.concatenate([position, position - state.gripper_position]).astype(np.float32)
                for name, position in state.block_positions.items()
            },
        }

    @classmethod
    def count_blocks(cls, observation_size: int) -> int:
        extra = observation_size - cls.BASE_OBSERVATION_SIZE
        if extra < 0 or extra % cls.EXTRA_BLOCK_SIZE != 0:
            raise ValueError(
                f"observation size {observation_size} is not a Fetch object task layout "
                f"({cls.BASE_OBSERVATION_SIZE} + {cls.EXTRA_BLOCK_SIZE} per extra block)"
            )
        return 1 + extra // cls.EXTRA_BLOCK_SIZE

    @classmethod
    def extract_state(cls, obs: dict) -> FetchState:
        raw = obs["observation"]
        block_positions = {"block0": raw[3:6].copy()}
        block_rotations = {"block0": raw[11:14].copy()}
        for i in range(1, cls.count_blocks(raw.shape[0])):
            start = cls.BASE_OBSERVATION_SIZE + cls.EXTRA_BLOCK_SIZE * (i - 1)
            block_positions[f"block{i}"] = raw[start : start + 3].copy()
            block_rotations[f"block{i}"] = raw[start + 6 : start + 9].copy()
        return FetchState(
            gripper_position=raw[0:3].copy(),
            finger_width=float(raw[9] + raw[10]),
            block_positions=block_positions,
            block_rotations=block_rotations,
        )
