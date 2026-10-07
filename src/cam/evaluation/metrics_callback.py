"""Stable-Baselines3 callback that evaluates the policy during training and records learning curves.

Evaluates at the start of training (zero-shot), every eval_interval environment
steps, and at the end (skipped if the last evaluation was at that step). Each
evaluation runs the policy deterministically and stochastically on a separate
evaluation environment, so training episodes are not interrupted, and its
results are:

    added, with the training experience so far (steps and episodes, in total and
        per skill), to that mode's LearningCurve (self.curves["deterministic" / "stochastic"])
    logged to TensorBoard under eval_<mode>/<skill>/...
    logged as one INFO line per skill
    written, with the derived metrics, to <run_directory>/metrics.json

Evaluation episodes are not counted as training steps. Every evaluation reseeds
the evaluation environment with the same seed, so all points on a curve are
measured on the same scenes.

Stopping: training stops after an evaluation in which every skill in stop_skills
has reached success_threshold (deterministic), or, with patience > 0, after
`patience` evaluations without improvement of the mean deterministic success rate
over all skills. The reason is logged and written to metrics.json.

Resuming: pass the run's metrics.json contents as `metrics`; curves and training
experience counters continue from it. write_metrics() also stores the current
counters ("progress"), so saving it with each checkpoint keeps them consistent
with the model.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from cam.evaluation.evaluation import evaluate
from cam.evaluation.learning_curve import MEASURES, LearningCurve, TrainingProgress
from cam.policies.stable_baselines3_policy import StableBaselines3Policy
from cam.training.checkpointing import atomic_write_text

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
        stop_skills: list[str] | None = None,
        patience: int = 0,
        metrics: dict | None = None,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.skill_names = skill_names
        self.eval_interval = eval_interval
        self.episodes_per_skill = episodes_per_skill
        self.run_directory = Path(run_directory)
        self.success_threshold = success_threshold
        self.seed = seed
        self.stop_skills = list(stop_skills or [])
        unknown = [skill for skill in self.stop_skills if skill not in skill_names]
        if unknown:
            raise ValueError(f"stop skills {unknown} are not among the trained skills {skill_names}")
        self.patience = patience
        self.best_mean_success = -1.0
        self.evaluations_without_improvement = 0
        self.stop_reason: str | None = None
        self.next_evaluation_step = 0
        if metrics:
            self.curves = {mode: LearningCurve.from_dict(metrics["curves"][mode]) for mode in MODES}
            saved = metrics.get("progress") or asdict(TrainingProgress(0, 0, {}, {}))
            self._restore_progress(TrainingProgress(**saved))
        else:
            self.curves = {mode: LearningCurve() for mode in MODES}
            self._restore_progress(TrainingProgress(0, 0, {}, {}))

    def _restore_progress(self, progress: TrainingProgress) -> None:
        self.saved_progress = progress
        self.episodes = progress.episodes
        self.skill_steps = dict(progress.skill_steps)
        self.skill_episodes = dict(progress.skill_episodes)

    def _on_training_start(self) -> None:
        self._reconcile_with_model()
        self._evaluate()
        self.next_evaluation_step = self.num_timesteps + self.eval_interval

    def _reconcile_with_model(self) -> None:
        """On resume, drop curve points after the model's step (written after its last checkpoint) and,
        if the saved counters are not at the model's step, fall back to the last point's counters."""
        step = self.num_timesteps
        for curve in self.curves.values():
            curve.points = [point for point in curve.points if point[0] <= step]
        if self.saved_progress.steps != step:
            points = self.curves["deterministic"].points
            fallback = points[-1][2] if points and points[-1][2] else TrainingProgress(step, 0, {}, {})
            logger.warning(
                "saved training counters are at step %d but the model is at step %d; using the counters of the "
                "last evaluation (step %d), so experience after it is not counted",
                self.saved_progress.steps, step, fallback.steps,
            )
            self._restore_progress(fallback)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            skill = info["skill"]
            self.skill_steps[skill] = self.skill_steps.get(skill, 0) + 1
            if done:
                self.episodes += 1
                self.skill_episodes[skill] = self.skill_episodes.get(skill, 0) + 1
        if self.num_timesteps >= self.next_evaluation_step:
            self._evaluate()
            self.next_evaluation_step += self.eval_interval
        return self.stop_reason is None

    def _on_training_end(self) -> None:
        if self.curves["deterministic"].last_step != self.num_timesteps:
            self._evaluate()

    def _evaluate(self) -> None:
        step = self.num_timesteps
        progress = TrainingProgress(step, self.episodes, dict(self.skill_steps), dict(self.skill_episodes))
        self.logger.record("progress/episodes", progress.episodes)
        for skill in self.skill_names:
            self.logger.record(f"progress/{skill}/steps", progress.skill_steps.get(skill, 0))
            self.logger.record(f"progress/{skill}/episodes", progress.skill_episodes.get(skill, 0))
        for mode, deterministic in MODES.items():
            policy = StableBaselines3Policy(self.model, deterministic=deterministic)
            result = evaluate(self.eval_env, policy, self.skill_names, self.episodes_per_skill, seed=self.seed)
            curve = self.curves[mode]
            curve.add(step, result, progress)
            for skill, evaluation in result.skills.items():
                prefix = f"eval_{mode}/{skill}"
                self.logger.record(f"{prefix}/success_rate", evaluation.success_rate)
                self.logger.record(f"{prefix}/success_rate_std", evaluation.success_rate_std)
                self.logger.record(f"{prefix}/return", evaluation.mean_return)
                self.logger.record(f"{prefix}/return_std", evaluation.return_std)
                self.logger.record(f"{prefix}/episode_length", evaluation.mean_episode_length)
                self.logger.record(f"{prefix}/episode_length_std", evaluation.episode_length_std)
                self.logger.record(f"{prefix}/auc", curve.auc(skill))
                for measure in MEASURES:
                    to_threshold = curve.steps_to_threshold(skill, self.success_threshold, measure=measure)
                    if to_threshold is not None:
                        self.logger.record(f"{prefix}/{measure}_to_threshold", to_threshold)
                logger.info(
                    "eval step %d %s %s: success %.2f ± %.2f, return %.2f ± %.2f, length %.1f ± %.1f (%d episodes)",
                    step, mode, skill, evaluation.success_rate, evaluation.success_rate_std,
                    evaluation.mean_return, evaluation.return_std,
                    evaluation.mean_episode_length, evaluation.episode_length_std, evaluation.episodes,
                )
        self.logger.dump(step)
        self._check_stop()
        self.write_metrics()

    def _check_stop(self) -> None:
        curve = self.curves["deterministic"]
        if self.stop_skills and all(
            curve.steps_to_threshold(skill, self.success_threshold) is not None for skill in self.stop_skills
        ):
            self.stop_reason = f"{', '.join(self.stop_skills)} reached success rate {self.success_threshold}"
        if self.patience > 0:
            _, result, _ = curve.points[-1]
            mean_success = float(np.mean([result.skills[skill].success_rate for skill in self.skill_names]))
            if mean_success > self.best_mean_success:
                self.best_mean_success, self.evaluations_without_improvement = mean_success, 0
            else:
                self.evaluations_without_improvement += 1
                if self.evaluations_without_improvement >= self.patience:
                    self.stop_reason = (
                        f"mean deterministic success rate has not improved for {self.patience} evaluations"
                    )
        if self.stop_reason:
            logger.info("stopping training at step %d: %s", self.num_timesteps, self.stop_reason)

    def summary(self) -> dict:
        """Per mode and skill: zero-shot success rate; training steps, episodes, skill steps and skill
        episodes to threshold; AUC; final success rate."""
        return {
            mode: {
                skill: {
                    "zero_shot_success_rate": curve.zero_shot(skill).success_rate,
                    **{
                        f"{measure}_to_threshold": curve.steps_to_threshold(
                            skill, self.success_threshold, measure=measure
                        )
                        for measure in MEASURES
                    },
                    "auc": curve.auc(skill),
                    "final_success_rate": curve.final_success_rate(skill),
                }
                for skill in curve.skills()
            }
            for mode, curve in self.curves.items()
        }

    def write_metrics(self) -> None:
        self.run_directory.mkdir(parents=True, exist_ok=True)
        progress = TrainingProgress(self.num_timesteps, self.episodes, dict(self.skill_steps), dict(self.skill_episodes))
        metrics = {
            "success_threshold": self.success_threshold,
            "episodes_per_skill": self.episodes_per_skill,
            "stop_reason": self.stop_reason,
            "progress": asdict(progress),
            "summary": self.summary(),
            "curves": {mode: curve.to_dict() for mode, curve in self.curves.items()},
        }
        atomic_write_text(self.run_directory / "metrics.json", json.dumps(metrics, indent=2))
