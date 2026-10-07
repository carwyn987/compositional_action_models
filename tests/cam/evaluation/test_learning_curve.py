import pytest

from cam.evaluation.evaluation import EvaluationResult, SkillEvaluation
from cam.evaluation.learning_curve import LearningCurve, TrainingProgress


def result(**success_rates: float) -> EvaluationResult:
    """An EvaluationResult with the given success rate per skill (other fields fixed)."""
    return EvaluationResult({skill: SkillEvaluation(10, rate, 0.0, rate, 0.0, 50.0, 0.0) for skill, rate in success_rates.items()})


def curve(*points) -> LearningCurve:
    learning_curve = LearningCurve()
    for step, evaluation in points:
        learning_curve.add(step, evaluation)
    return learning_curve


@pytest.mark.unit
def test_steps_to_threshold_counts_from_the_first_evaluation():
    """First evaluation at or above the threshold, as steps since the skill's zero-shot point."""
    c = curve((0, result(a=0.0)), (100, result(a=0.5)), (200, result(a=0.96)), (300, result(a=1.0)))
    assert c.steps_to_threshold("a", threshold=0.95) == 200
    assert c.steps_to_threshold("a", threshold=0.99) == 300


@pytest.mark.unit
def test_consecutive_evaluations_must_stay_at_threshold():
    """With consecutive=2 a single high evaluation followed by a drop does not count; the streak's start does."""
    c = curve((0, result(a=0.0)), (100, result(a=1.0)), (200, result(a=0.4)), (300, result(a=1.0)), (400, result(a=1.0)))
    assert c.steps_to_threshold("a", threshold=0.95, consecutive=2) == 300
    assert c.steps_to_threshold("a", threshold=0.95, consecutive=3) is None


@pytest.mark.unit
def test_skill_added_later_is_measured_from_its_own_first_evaluation():
    """A skill first evaluated at step 1000 (added on resume) has its zero-shot point and step count from there."""
    c = curve((0, result(a=0.2)), (1000, result(a=1.0, b=0.1)), (1500, result(a=1.0, b=1.0)))
    assert c.zero_shot("b").success_rate == 0.1
    assert c.steps_to_threshold("b") == 500
    assert c.skills() == ["a", "b"]


@pytest.mark.unit
def test_auc_is_mean_success_over_the_span():
    """Trapezoidal area under success vs. steps divided by the span: a linear rise from 0 to 1 gives 0.5."""
    assert curve((0, result(a=0.0)), (100, result(a=0.5)), (200, result(a=1.0))).auc("a") == pytest.approx(0.5)
    assert curve((0, result(a=1.0)), (100, result(a=1.0))).auc("a") == pytest.approx(1.0)
    assert curve((0, result(a=0.3))).auc("a") == pytest.approx(0.3)


@pytest.mark.unit
def test_final_success_rate_averages_the_last_evaluations():
    c = curve((0, result(a=0.0)), (100, result(a=0.6)), (200, result(a=0.8)))
    assert c.final_success_rate("a") == pytest.approx(0.8)
    assert c.final_success_rate("a", last_k=2) == pytest.approx(0.7)


@pytest.mark.unit
def test_metrics_of_an_unevaluated_skill_are_none():
    c = curve((0, result(a=0.5)))
    assert (c.zero_shot("z"), c.steps_to_threshold("z"), c.auc("z"), c.final_success_rate("z")) == (None, None, None, None)


@pytest.mark.unit
def test_steps_must_not_go_back():
    """Points are in step order: an earlier step is an error; the same step replaces the last point."""
    c = curve((100, result(a=0.5)))
    with pytest.raises(ValueError):
        c.add(50, result(a=0.6))
    c.add(100, result(a=0.7, b=0.1))
    assert len(c.points) == 1 and c.zero_shot("b").success_rate == 0.1


@pytest.mark.unit
def test_curve_round_trips_through_a_dict():
    """to_dict / from_dict (used for metrics.json and resuming) preserve every point."""
    c = curve((0, result(a=0.0)), (100, result(a=0.5, b=0.2)))
    assert LearningCurve.from_dict(c.to_dict()) == c


@pytest.mark.unit
def test_threshold_in_episodes_and_per_skill_experience():
    """The same threshold point measured as total episodes, or as the skill's own steps / episodes,
    counted from the skill's zero-shot point."""
    c = LearningCurve()
    c.add(0, result(a=0.0), TrainingProgress(0, 0, {}, {}))
    c.add(100, result(a=0.5, b=0.0), TrainingProgress(100, 4, {"a": 100}, {"a": 4}))
    c.add(300, result(a=1.0, b=1.0), TrainingProgress(300, 12, {"a": 220, "b": 80}, {"a": 8, "b": 4}))
    assert c.steps_to_threshold("a", measure="episodes") == 12
    assert c.steps_to_threshold("a", measure="skill_steps") == 220
    assert c.steps_to_threshold("b", measure="steps") == 200  # b's zero-shot point is step 100
    assert c.steps_to_threshold("b", measure="skill_episodes") == 4
    assert LearningCurve.from_dict(c.to_dict()) == c
