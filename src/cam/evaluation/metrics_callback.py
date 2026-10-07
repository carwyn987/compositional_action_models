"""Stable-Baselines3 callback that evaluates the policy during training and records learning curves.

Evaluates at the start of training (zero-shot), every eval_interval environment
steps, and at the end (skipped if the last evaluation was at that step). Each
evaluation runs the policy deterministically and stochastically on a separate
evaluation environment, so training episodes are not interrupted, and its
results are:

    added to that mode's LearningCurve (self.curves["deterministic" / "stochastic"])
    logged to TensorBoard under eval_<mode>/<skill>/...
    logged as one INFO line per skill
    written, with the derived metrics, to <run_directory>/metrics.json

Evaluation episodes are not counted as training steps. Every evaluation reseeds
the evaluation environment with the same seed, so all points on a curve are
measured on the same scenes.
"""

import json
import logging
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

from cam.evaluation.evaluation import evaluate
from cam.evaluation.learning_curve import LearningCurve
from cam.policies.stable_baselines3_policy import StableBaselines3Policy

logger = logging.getLogger(__name__)

MODES = {"deterministic": True, "stochastic": False}


class MetricsCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        skill_names: list[str],
        eval_interval: int,
        episodes_per_skill: int,
        run_directory: Path,
        success_threshold: float = 0.95,
        seed: int = 0,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.skill_names = skill_names
        self.eval_interval = eval_interval
        self.episodes_per_skill = episodes_per_skill
        self.run_directory = Path(run_directory)
        self.success_threshold = success_threshold
        self.seed = seed
        self.curves = {mode: LearningCurve() for mode in MODES}
        self.next_evaluation_step = 0

    def _on_training_start(self) -> None:
        self._evaluate()
        self.next_evaluation_step = self.num_timesteps + self.eval_interval

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_evaluation_step:
            self._evaluate()
            self.next_evaluation_step += self.eval_interval
        return True

    def _on_training_end(self) -> None:
        if self.curves["deterministic"].last_step != self.num_timesteps:
            self._evaluate()

    def _evaluate(self) -> None:
        step = self.num_timesteps
        for mode, deterministic in MODES.items():
            policy = StableBaselines3Policy(self.model, deterministic=deterministic)
            result = evaluate(self.eval_env, policy, self.skill_names, self.episodes_per_skill, seed=self.seed)
            curve = self.curves[mode]
            curve.add(step, result)
            for skill, evaluation in result.skills.items():
                prefix = f"eval_{mode}/{skill}"
                self.logger.record(f"{prefix}/success_rate", evaluation.success_rate)
                self.logger.record(f"{prefix}/success_rate_std", evaluation.success_rate_std)
                self.logger.record(f"{prefix}/return", evaluation.mean_return)
                self.logger.record(f"{prefix}/return_std", evaluation.return_std)
                self.logger.record(f"{prefix}/episode_length", evaluation.mean_episode_length)
                self.logger.record(f"{prefix}/episode_length_std", evaluation.episode_length_std)
                self.logger.record(f"{prefix}/auc", curve.auc(skill))
                steps_to_threshold = curve.steps_to_threshold(skill, self.success_threshold)
                if steps_to_threshold is not None:
                    self.logger.record(f"{prefix}/steps_to_threshold", steps_to_threshold)
                logger.info(
                    "eval step %d %s %s: success %.2f ± %.2f, return %.2f ± %.2f, length %.1f ± %.1f (%d episodes)",
                    step, mode, skill, evaluation.success_rate, evaluation.success_rate_std,
                    evaluation.mean_return, evaluation.return_std,
                    evaluation.mean_episode_length, evaluation.episode_length_std, evaluation.episodes,
                )
        self.logger.dump(step)
        self.write_metrics()

    def summary(self) -> dict:
        """Per mode and skill: zero-shot success rate, steps to threshold, AUC, final success rate."""
        return {
            mode: {
                skill: {
                    "zero_shot_success_rate": curve.zero_shot(skill).success_rate,
                    "steps_to_threshold": curve.steps_to_threshold(skill, self.success_threshold),
                    "auc": curve.auc(skill),
                    "final_success_rate": curve.final_success_rate(skill),
                }
                for skill in curve.skills()
            }
            for mode, curve in self.curves.items()
        }

    def write_metrics(self) -> None:
        self.run_directory.mkdir(parents=True, exist_ok=True)
        metrics = {
            "success_threshold": self.success_threshold,
            "episodes_per_skill": self.episodes_per_skill,
            "summary": self.summary(),
            "curves": {mode: curve.to_dict() for mode, curve in self.curves.items()},
        }
        (self.run_directory / "metrics.json").write_text(json.dumps(metrics, indent=2))
