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
has reached success_threshold (deterministic) on stop_consecutive evaluations in a
row, or, with patience > 0, after `patience` evaluations without improvement of the
mean deterministic success rate over all skills. With recover_others, the stop
skills must be at the threshold now and every other skill must also be back at its
level from before the most recently added skills (see retention()). The reason is
logged and written to metrics.json.

Retention: when skills were added after training started (e.g. a repaired operator
replacing its original), metrics.json also records, for every other skill, its
level before the addition, its lowest and last success rate after it, and when it
recovered (retention()): how much the new skill made the policy forget.

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

from cam.evaluation.evaluation import EvaluationResult, evaluate
from cam.evaluation.learning_curve import MEASURES, LearningCurve, TrainingProgress
from cam.policies.stable_baselines3_policy import StableBaselines3Policy
from cam.training.checkpointing import atomic_write_text

logger = logging.getLogger(__name__)

MODES = {"deterministic": True, "stochastic": False}
# Skills added on resume (e.g. a repaired operator) are also evaluated alone this many steps after being added,
# doubling (1k, 2k, 4k, ...) until the regular --eval-interval takes over: precise steps to threshold early,
# where conditions differ most, at little cost.
ADDED_SKILL_FIRST_EVALUATION = 1_000
# A skill's level before skills were added: the mean of its last this many evaluations up to the addition, so one
# lucky or unlucky evaluation (20 episodes: about ±0.1) does not set it.
LEVEL_EVALUATIONS = 2


def added_at(curve: LearningCurve) -> int | None:
    """The step at which the most recently added skills were first evaluated (their zero-shot point); None if
    every skill was evaluated from the curve's first point."""
    first_seen: dict[str, int] = {}
    for step, result, _ in curve.points:
        for skill in result.skills:
            first_seen.setdefault(skill, step)
    later = [step for step in first_seen.values() if step > curve.points[0][0]]
    return max(later) if later else None


def holds(rates: list[float], bound: float, consecutive: int) -> bool:
    """The last `consecutive` success rates are all at least bound."""
    return len(rates) >= consecutive and all(rate >= bound - 1e-9 for rate in rates[-consecutive:])


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
        stop_consecutive: int = 1,
        recover_others: bool = False,
        recovery_tolerance: float = 0.1,
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
        self.stop_consecutive = stop_consecutive  # evaluations in a row at the threshold (and recovered) to stop
        self.recover_others = recover_others  # also wait until every other skill is back at its earlier level
        self.recovery_tolerance = recovery_tolerance  # how far below its earlier level a skill counts as back
        self.added_skills: list[str] = []  # skills added on resume, evaluated on the dense-then-sparse schedule
        self.added_at = 0
        self.next_added_evaluation = ADDED_SKILL_FIRST_EVALUATION  # steps after added_at
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
        points = self.curves["deterministic"].points
        seen = {skill for _, result, _ in points for skill in result.skills}
        self.added_skills = [skill for skill in self.skill_names if points and skill not in seen]
        self.added_at = self.num_timesteps
        if points and points[-1][0] == self.num_timesteps:
            # Resumed right at the last evaluation: evaluate only skills added since (their zero-shot point).
            missing = [skill for skill in self.skill_names if skill not in points[-1][1].skills]
            if missing:
                self._evaluate(missing)
        else:
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
        added_offset = self.num_timesteps - self.added_at
        if (self.added_skills and self.next_added_evaluation < self.eval_interval
                and added_offset >= self.next_added_evaluation):
            if self.curves["deterministic"].last_step != self.num_timesteps:
                self._evaluate(self.added_skills)  # dense early: 1k, 2k, 4k, ... after the skills were added
            self.next_added_evaluation *= 2
        return self.stop_reason is None

    def _on_training_end(self) -> None:
        # A stop follows right after the evaluation that triggered it, so that evaluation is the last point.
        if self.stop_reason is None and self.curves["deterministic"].last_step != self.num_timesteps:
            self._evaluate()

    def _evaluate(self, skill_names: list[str] | None = None) -> None:
        """Evaluate skill_names (default: all skills). Evaluating a subset at the last point's step adds
        its results to that point instead of re-evaluating the others."""
        skill_names = skill_names or self.skill_names
        step = self.num_timesteps
        progress = TrainingProgress(step, self.episodes, dict(self.skill_steps), dict(self.skill_episodes))
        self.logger.record("progress/episodes", progress.episodes)
        for skill in self.skill_names:
            self.logger.record(f"progress/{skill}/steps", progress.skill_steps.get(skill, 0))
            self.logger.record(f"progress/{skill}/episodes", progress.skill_episodes.get(skill, 0))
        for mode, deterministic in MODES.items():
            policy = StableBaselines3Policy(self.model, deterministic=deterministic)
            result = evaluate(self.eval_env, policy, skill_names, self.episodes_per_skill, seed=self.seed)
            curve = self.curves[mode]
            if curve.points and curve.points[-1][0] == step:
                result = EvaluationResult(curve.points[-1][1].skills | result.skills)
            curve.add(step, result, progress)
            for skill in skill_names:  # only the skills just evaluated
                evaluation = result.skills[skill]
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
        reached = f"{', '.join(self.stop_skills)} reached success rate {self.success_threshold}" + (
            f" on {self.stop_consecutive} evaluations in a row" if self.stop_consecutive > 1 else "")
        if self.stop_skills and all(
            curve.steps_to_threshold(skill, self.success_threshold, consecutive=self.stop_consecutive) is not None
            for skill in self.stop_skills
        ):
            if not self.recover_others:
                self.stop_reason = reached
            else:  # the stop skills at the threshold now, and every other skill back at its earlier level
                at_threshold = all(
                    holds([e.success_rate for _, e in curve.evaluations(skill)], self.success_threshold,
                          self.stop_consecutive)
                    for skill in self.stop_skills
                )
                retention = self.retention()
                if at_threshold and retention and all(entry["recovered"] for entry in retention["skills"].values()):
                    self.stop_reason = f"{reached}, and every other skill is back at its level before it was added"
        if self.patience > 0 and all(skill in curve.points[-1][1].skills for skill in self.skill_names):
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

    def retention(self) -> dict | None:
        """How the skills trained before the most recent addition fared after it (None without an addition):

            added_at              the step the skills were added (their zero-shot point)
            skills[skill]         for each other current skill:
                level_before      mean of its last LEVEL_EVALUATIONS evaluations up to added_at
                lowest_after      its lowest success rate after added_at (None before its first evaluation)
                final             its last success rate after added_at
                recovered_at      the first step after added_at at which its last stop_consecutive evaluations
                                  since added_at were all at least level_before - recovery_tolerance
                recovered         whether that holds now
            all_recovered_at      the first step at which all of them held it at once

        Deterministic evaluations. Only evaluations after added_at count toward recovery, so a skill is never
        recovered before it was evaluated again."""
        curve = self.curves["deterministic"]
        step_added = added_at(curve) if curve.points else None
        if step_added is None:
            return None
        rates = {skill: [(step, e.success_rate) for step, e in curve.evaluations(skill)] for skill in self.skill_names}
        skills, bounds = {}, {}
        for skill, evaluations in rates.items():
            before = [rate for step, rate in evaluations if step <= step_added]
            after = [(step, rate) for step, rate in evaluations if step > step_added]
            if not before or evaluations[0][0] == step_added:
                continue  # one of the added skills
            level = float(np.mean(before[-LEVEL_EVALUATIONS:]))
            bounds[skill] = level - self.recovery_tolerance
            after_rates = [rate for _, rate in after]
            skills[skill] = {
                "level_before": level,
                "lowest_after": min(after_rates) if after else None,
                "final": after_rates[-1] if after else None,
                "recovered_at": next(
                    (step for i, (step, _) in enumerate(after)
                     if holds(after_rates[: i + 1], bounds[skill], self.stop_consecutive)), None),
                "recovered": holds(after_rates, bounds[skill], self.stop_consecutive),
            }
        steps = sorted({step for skill in skills for step, _ in rates[skill] if step > step_added})
        all_recovered_at = next(
            (step for step in steps if all(
                holds([rate for s, rate in rates[skill] if step_added < s <= step], bounds[skill], self.stop_consecutive)
                for skill in skills)),
            None,
        ) if skills else None
        return {"added_at": step_added, "skills": skills, "all_recovered_at": all_recovered_at}

    def write_metrics(self) -> None:
        self.run_directory.mkdir(parents=True, exist_ok=True)
        progress = TrainingProgress(self.num_timesteps, self.episodes, dict(self.skill_steps), dict(self.skill_episodes))
        metrics = {
            "success_threshold": self.success_threshold,
            "episodes_per_skill": self.episodes_per_skill,
            "stop_reason": self.stop_reason,
            "progress": asdict(progress),
            "summary": self.summary(),
            "retention": self.retention(),
            "curves": {mode: curve.to_dict() for mode, curve in self.curves.items()},
        }
        atomic_write_text(self.run_directory / "metrics.json", json.dumps(metrics, indent=2))
