"""Train a policy with Stable-Baselines3 (SAC or PPO) on the policy observation.

The environment is the full stack from main.setup_environment, whose
observations are {"observation", "operator_embedding", "grounding"}; SB3's
MultiInputPolicy concatenates them and feeds an MLP. One model is trained on
all skills: SkillEnvironment chooses the episode's skill, and the operator
embedding and grounding tell the policy which skill and objects it is acting on.

Outputs, under <output_dir>/<run_name>/:
    model.zip          the trained SB3 model
    monitor.csv        per-episode return, length, is_success, skill
    tensorboard/       SB3 training curves, plus success_rate/<skill>

Hyperparameters follow deprecated/train_skill.py.
"""

import logging
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

logger = logging.getLogger(__name__)

ALGORITHMS = {"sac": SAC, "ppo": PPO}
HYPERPARAMETERS = {
    "sac": dict(
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=1_000,
        batch_size=256,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
    ),
    "ppo": dict(learning_rate=3e-4, n_steps=1024, batch_size=256, gamma=0.95),
}


class SuccessRateCallback(BaseCallback):
    """Logs each skill's success rate over its last `window` episodes as success_rate/<skill>.

    Reads info["skill"] and info["is_success"] from SkillEnvironment at the end of
    each episode. SB3 also logs rollout/success_rate across all skills.
    """

    def __init__(self, window: int = 100):
        super().__init__()
        self.window = window
        self.outcomes: dict[str, deque] = {}

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                outcomes = self.outcomes.setdefault(info["skill"], deque(maxlen=self.window))
                outcomes.append(float(info["is_success"]))
                self.logger.record(f"success_rate/{info['skill']}", float(np.mean(outcomes)))
        return True


def train_stable_baselines3(
    env: gym.Env,
    algorithm: str,
    total_timesteps: int,
    run_directory: Path,
    seed: int = 0,
    hyperparameter_overrides: dict | None = None,
) -> BaseAlgorithm:
    """Train an SB3 model on env for total_timesteps environment steps and save it to run_directory."""
    run_directory.mkdir(parents=True, exist_ok=True)
    monitored_env = Monitor(env, filename=str(run_directory), info_keywords=("is_success", "skill"))
    hyperparameters = HYPERPARAMETERS[algorithm] | (hyperparameter_overrides or {})
    model = ALGORITHMS[algorithm](
        "MultiInputPolicy",
        monitored_env,
        seed=seed,
        verbose=1,
        tensorboard_log=str(run_directory / "tensorboard"),
        **hyperparameters,
    )
    logger.info("training %s for %d steps; outputs in %s", algorithm, total_timesteps, run_directory)
    model.learn(total_timesteps=total_timesteps, callback=SuccessRateCallback(), progress_bar=True)
    model.save(run_directory / "model")
    logger.info("saved model to %s", run_directory / "model.zip")
    return model
