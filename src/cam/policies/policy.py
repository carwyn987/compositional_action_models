"""Policy interface: chooses an action for a grounded action in the current state."""

from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np

from cam.domain.symbolic_action_model import GroundedSymbolicActionModel


class Policy(ABC):
    def reset(self) -> None:
        """Called before each execution of a grounded action."""

    @abstractmethod
    def __call__(self, obs, info: dict, grounded_action_model: GroundedSymbolicActionModel) -> np.ndarray:
        """Action to take now, given the observation, info and the action being executed."""


class RandomPolicy(Policy):
    def __init__(self, action_space: gym.Space):
        self.action_space = action_space

    def __call__(self, obs, info, grounded_action_model) -> np.ndarray:
        return self.action_space.sample()
