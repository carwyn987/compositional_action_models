"""Gymnasium-Robotics Fetch environments annotated with a named state."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass(frozen=True, eq=False)
class FetchState:
    gripper_position: np.ndarray  # (3,) x, y, z
    finger_width: float  # sum of the two finger joint positions
    block_positions: dict[str, np.ndarray]  # block name -> (3,) x, y, z


class FetchEnvStateAnnotationWrapper(gym.Wrapper):
    """Adds info["environment_state"], a FetchState of the current observation.

    Works for the stock single-block Fetch object tasks and FetchMultiBlock-v0.
    Observations are passed through unchanged. Layout of obs["observation"]:
        0:3    gripper position
        3:6    block0 position
        9:11   finger joint positions
        25+9k  block k+1 position (3), relative position (3), rotation (3)
    Blocks are named block0..block{N-1}; N is determined by the observation length.
    """

    BASE_OBSERVATION_SIZE = 25
    EXTRA_BLOCK_SIZE = 9

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.num_blocks = self.count_blocks(env.observation_space["observation"].shape[0])

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, {**info, "environment_state": self.extract_state(obs)}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = {**info, "environment_state": self.extract_state(obs)}
        return obs, reward, terminated, truncated, info

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
        for i in range(1, cls.count_blocks(raw.shape[0])):
            start = cls.BASE_OBSERVATION_SIZE + cls.EXTRA_BLOCK_SIZE * (i - 1)
            block_positions[f"block{i}"] = raw[start : start + 3].copy()
        return FetchState(
            gripper_position=raw[0:3].copy(),
            finger_width=float(raw[9] + raw[10]),
            block_positions=block_positions,
        )
