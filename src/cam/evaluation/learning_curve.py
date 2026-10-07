"""Learning curves: evaluation results over training, and the metrics derived from them.

A LearningCurve stores (environment step, EvaluationResult, TrainingProgress)
points in step order; TrainingProgress records the training experience so far,
in total and per skill. Each skill's metrics are measured from that skill's first evaluation,
which is its zero-shot point: the start of training for skills trained from the
beginning, or the step a skill was added for skills added later (new or
repaired skills when resuming).

    zero_shot(skill)                              first evaluation of the skill
    steps_to_threshold(skill, threshold, k, m)    training experience from the zero-shot point until an
                                                  evaluation's success rate is at or above threshold
                                                  (for k evaluations in a row; k = 1 by default), measured
                                                  in m: steps, episodes, skill_steps or skill_episodes
    auc(skill)                                    mean success rate over the steps evaluated so far
                                                  (trapezoidal area under success vs. steps / span), in [0, 1]
    final_success_rate(skill, last_k)             mean success rate of the last k evaluations

Each point's success rate is over that evaluation's episodes (episodes_per_skill),
so "success >= 0.95 over 20 consecutive episodes" is one evaluation of 20
episodes reaching the threshold: steps_to_threshold(skill, 0.95) with 20
evaluation episodes.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from cam.evaluation.evaluation import EvaluationResult, SkillEvaluation


MEASURES = ("steps", "episodes", "skill_steps", "skill_episodes")


@dataclass(frozen=True)
class TrainingProgress:
    """Training experience when an evaluation was taken: environment steps and completed episodes,
    in total and per skill (setup steps are not training experience and are not counted)."""

    steps: int
    episodes: int
    skill_steps: dict[str, int]
    skill_episodes: dict[str, int]

    def measure(self, measure: str, skill: str) -> int:
        """steps / episodes in total, or skill_steps / skill_episodes of the given skill."""
        if measure in ("steps", "episodes"):
            return getattr(self, measure)
        return getattr(self, measure).get(skill, 0)


@dataclass
class LearningCurve:
    # (training step of the evaluation, its result, training experience up to it); progress may be None
    points: list[tuple[int, EvaluationResult, TrainingProgress | None]] = field(default_factory=list)

    def add(self, step: int, result: EvaluationResult, progress: TrainingProgress | None = None) -> None:
        if self.points and step <= self.points[-1][0]:
            raise ValueError(f"step {step} is not after the last evaluated step {self.points[-1][0]}")
        self.points.append((step, result, progress))

    @property
    def last_step(self) -> int | None:
        return self.points[-1][0] if self.points else None

    def skills(self) -> list[str]:
        """Skills evaluated at least once, in order of first evaluation."""
        names: dict[str, None] = {}
        for _, result, _ in self.points:
            names.update(dict.fromkeys(result.skills))
        return list(names)

    def evaluations(self, skill: str) -> list[tuple[int, SkillEvaluation]]:
        """(step, evaluation) for every point that evaluated the skill."""
        return [(step, result.skills[skill]) for step, result, _ in self.points if skill in result.skills]

    def success_rates(self, skill: str) -> tuple[np.ndarray, np.ndarray]:
        """(steps, success rates) of the skill's evaluations."""
        evaluations = self.evaluations(skill)
        return (
            np.array([step for step, _ in evaluations], dtype=np.int64),
            np.array([evaluation.success_rate for _, evaluation in evaluations], dtype=np.float64),
        )

    def zero_shot(self, skill: str) -> SkillEvaluation | None:
        evaluations = self.evaluations(skill)
        return evaluations[0][1] if evaluations else None

    def steps_to_threshold(
        self, skill: str, threshold: float = 0.95, consecutive: int = 1, measure: str = "steps"
    ) -> int | None:
        """Training experience from the skill's first evaluation until an evaluation has success
        rate >= threshold; None if not reached (yet).

        The success rate is over the evaluation's episodes, so with 20 evaluation episodes and
        threshold 0.95 this is the first evaluation in which at least 19 of 20 episodes succeeded.
        consecutive > 1 additionally requires that many evaluations in a row at or above the
        threshold, and measures up to the first of them.

        measure: "steps" or "episodes" of all training, or "skill_steps" / "skill_episodes" of
        this skill's own training experience (needs TrainingProgress on the points).
        """
        if measure not in MEASURES:
            raise ValueError(f"measure must be one of {MEASURES}, got {measure!r}")
        points = [(step, result.skills[skill].success_rate, progress)
                  for step, result, progress in self.points if skill in result.skills]
        streak = 0
        for i, (_, rate, _) in enumerate(points):
            streak = streak + 1 if rate >= threshold else 0
            if streak == consecutive:
                (start_step, _, start), (step, _, reached) = points[0], points[i - consecutive + 1]
                if measure == "steps":
                    return int(step - start_step)
                if start is None or reached is None:
                    raise ValueError(f"measure {measure!r} needs TrainingProgress on the curve's points")
                return reached.measure(measure, skill) - start.measure(measure, skill)
        return None

    def auc(self, skill: str) -> float | None:
        """Mean success rate over the evaluated span: area under success vs. steps divided by the span.
        A single evaluation gives its own success rate."""
        steps, rates = self.success_rates(skill)
        if len(steps) == 0:
            return None
        if len(steps) == 1:
            return float(rates[0])
        return float(np.trapezoid(rates, steps) / (steps[-1] - steps[0]))

    def final_success_rate(self, skill: str, last_k: int = 1) -> float | None:
        """Mean success rate of the skill's last `last_k` evaluations; None if never evaluated.

        "Final success rate" in docs/research_focus.md (RQ1). last_k > 1 averages
        out evaluation noise at the end of training.
        """
        _, rates = self.success_rates(skill)
        return float(np.mean(rates[-last_k:])) if len(rates) else None

    def to_dict(self) -> dict:
        return {
            "points": [
                {"step": step, "result": result.to_dict(), "progress": asdict(progress) if progress else None}
                for step, result, progress in self.points
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LearningCurve":
        return cls([
            (
                point["step"],
                EvaluationResult.from_dict(point["result"]),
                TrainingProgress(**point["progress"]) if point.get("progress") else None,
            )
            for point in data["points"]
        ])
