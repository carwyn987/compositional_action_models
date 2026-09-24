"""Fetch environments. Importing this package registers FetchMultiBlock-v0."""

import gymnasium as gym
import gymnasium_robotics

from cam.environments.fetch import fetch_multiblock_environment  # noqa: F401

gym.register_envs(gymnasium_robotics)
