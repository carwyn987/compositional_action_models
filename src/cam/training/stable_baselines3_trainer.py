"""Train a policy with Stable-Baselines3 (SAC or PPO) on the policy observation.

The environment is the full stack from main.setup_environment, whose
observations are {"observation", "operator_embedding", "grounding"}; SB3's
MultiInputPolicy concatenates them and feeds an MLP. One model is trained on
all skills: SkillEnvironment chooses the episode's skill, and the operator
embedding and grounding tell the policy which skill and objects it is acting on.

Outputs, under <output_dir>/<run_name>/:
    model.zip          the SB3 model: saved every save_interval steps, at the end, and on interruption
    monitor.csv        per-episode return, length, is_success, skill
    tensorboard/       SB3 training curves, plus success_rate/<skill> (and eval_<mode>/<skill>/... with a
                       MetricsCallback, which also writes metrics.json)

Hyperparameters follow deprecated/train_skill.py.
"""

import logging
from collections import deque
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from cam.training.checkpointing import PeriodicSaveCallback, atomic_save_model, atomic_save_replay_buffer

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
    callbacks: list[BaseCallback] | None = None,
    save_interval: int = 0,
    resume: bool = False,
    save_replay_buffer: bool = False,
    on_save: list[Callable[[], None]] | None = None,
    policy_kwargs: dict | None = None,
) -> BaseAlgorithm:
    """Train an SB3 model on env for total_timesteps environment steps and save it to run_directory.

    resume=True continues the model in run_directory/model.zip (and its replay_buffer.pkl, if
    saved) for total_timesteps more steps: the step count, TensorBoard curves and monitor.csv
    continue. callbacks run alongside SuccessRateCallback (e.g. a MetricsCallback).

    Saving (atomic) happens every save_interval steps (0: only at the end), at the end, and when
    training is interrupted (Ctrl+C) or raises, before the exception propagates. It writes
    model.zip, replay_buffer.pkl if save_replay_buffer (off-policy algorithms; can be large),
    and calls each on_save function (e.g. writing metrics).

    policy_kwargs go to a new model's policy (e.g. a features extractor); a resumed model keeps its own.
    """
    run_directory.mkdir(parents=True, exist_ok=True)
    model_path = run_directory / "model.zip"
    replay_buffer_path = run_directory / "replay_buffer.pkl"
    monitored_env = Monitor(
        env, filename=str(run_directory), info_keywords=("is_success", "skill"), override_existing=not resume
    )
    if resume:
        model = ALGORITHMS[algorithm].load(
            model_path, env=monitored_env, tensorboard_log=str(run_directory / "tensorboard")
        )
        if replay_buffer_path.exists() and hasattr(model, "load_replay_buffer"):
            model.load_replay_buffer(replay_buffer_path)
            logger.info("loaded replay buffer from %s", replay_buffer_path)
        logger.info("resuming %s from step %d", algorithm, model.num_timesteps)
    else:
        model = ALGORITHMS[algorithm](
            "MultiInputPolicy",
            monitored_env,
            seed=seed,
            verbose=1,
            tensorboard_log=str(run_directory / "tensorboard"),
            policy_kwargs=policy_kwargs,
            **(HYPERPARAMETERS[algorithm] | (hyperparameter_overrides or {})),
        )

    def save() -> None:
        atomic_save_model(model, model_path)
        if save_replay_buffer and getattr(model, "replay_buffer", None) is not None:
            atomic_save_replay_buffer(model, replay_buffer_path)
        for save_more in on_save or []:
            save_more()

    all_callbacks = [SuccessRateCallback(), *(callbacks or [])]
    if save_interval > 0:
        all_callbacks.append(PeriodicSaveCallback(save_interval, save))
    logger.info("training %s for %d steps; outputs in %s", algorithm, total_timesteps, run_directory)
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=all_callbacks,
            reset_num_timesteps=not resume,
            progress_bar=True,
        )
    except BaseException:
        save()
        logger.warning("training stopped at step %d; saved model to %s", model.num_timesteps, model_path)
        raise
    save()
    logger.info("training finished at step %d; saved model to %s", model.num_timesteps, model_path)
    return model
