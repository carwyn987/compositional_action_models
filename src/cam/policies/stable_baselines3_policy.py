"""A trained Stable-Baselines3 model used as a Policy."""

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from cam.policies.policy import Policy

ALGORITHMS = {"sac": SAC, "ppo": PPO}


class StableBaselines3Policy(Policy):
    """Acts with model.predict on the policy observation (see PolicyObservationWrapper)."""

    def __init__(self, model: BaseAlgorithm, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    @classmethod
    def load(cls, model_path: Path, algorithm: str, deterministic: bool = True) -> "StableBaselines3Policy":
        return cls(ALGORITHMS[algorithm].load(model_path), deterministic)

    def __call__(self, obs, info, grounded_action_model) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return action
