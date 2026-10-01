"""Reward functions: score each step of an episode of a grounded action.

SkillEnvironment decides success and termination from the action model; a
RewardFunction only turns a step into a number. Environment-specific shaped
rewards (e.g. environments/fetch/fetch_rewards.py) implement the same interface.
"""

from abc import ABC, abstractmethod

import numpy as np

from cam.domain.symbolic_action_model import GroundedSymbolicActionModel


class RewardFunction(ABC):
    def reset(self, grounded_action_model: GroundedSymbolicActionModel, info: dict) -> None:
        """Called at the start of each episode, after setup, with the episode's grounding."""

    @abstractmethod
    def __call__(
        self, grounded_action_model: GroundedSymbolicActionModel, action: np.ndarray, info: dict, success: bool
    ) -> tuple[float, dict[str, float]]:
        """(reward, components): the step's reward and its named terms (summing to it) for logging."""


class SparseReward(RewardFunction):
    """1.0 on the step the grounding's effects hold, else 0.0."""

    def __call__(self, grounded_action_model, action, info, success):
        reward = 1.0 if success else 0.0
        return reward, {"success": reward}
