import json

import pytest

pytest.importorskip("stable_baselines3")

from main import parse_args  # noqa: E402
from cam.evaluation.metrics_callback import MetricsCallback  # noqa: E402
from cam.experiments.environment_setup import setup_environment  # noqa: E402
from cam.skills.registry import build_skill  # noqa: E402
from cam.training.stable_baselines3_trainer import train_stable_baselines3  # noqa: E402
from tests.cam.evaluation.test_learning_curve import result  # noqa: E402


@pytest.mark.integration
def test_evaluates_at_start_interval_and_end_and_writes_metrics(tmp_path):
    """200 SAC steps with eval_interval 100: evaluations at steps 0 (zero-shot), 100 and 200, in both modes."""
    config = parse_args(["--skills", "pickup", "putdown", "--num-blocks", "2", "--max-steps-per-episode", "10"])
    skills = [build_skill(name, config) for name in config["skills"]]
    env, eval_env = setup_environment(config, skills), setup_environment(config, skills)
    callback = MetricsCallback(eval_env, ["pickup", "putdown"], 100, 1, tmp_path)
    try:
        train_stable_baselines3(
            env, "sac", 200, tmp_path, hyperparameter_overrides={"learning_starts": 50, "batch_size": 32},
            callbacks=[callback],
        )
    finally:
        env.close()
        eval_env.close()

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    for mode in ("deterministic", "stochastic"):
        assert [point["step"] for point in metrics["curves"][mode]["points"]] == [0, 100, 200]
        assert set(metrics["summary"][mode]) == {"pickup", "putdown"}
        assert set(metrics["summary"][mode]["pickup"]) == {
            "zero_shot_success_rate", "steps_to_threshold", "episodes_to_threshold",
            "skill_steps_to_threshold", "skill_episodes_to_threshold", "auc", "final_success_rate",
        }
    final = metrics["curves"]["deterministic"]["points"][-1]["progress"]
    assert final["steps"] == 200 and sum(final["skill_steps"].values()) == 200
    assert final["episodes"] == sum(final["skill_episodes"].values()) > 0


def callback_with_curve(success_rates_per_point, **kwargs):
    """A MetricsCallback whose deterministic curve holds the given points (no environment needed)."""
    callback = MetricsCallback(None, ["a", "b"], 100, 1, "unused", **kwargs)
    for step, rates in enumerate(success_rates_per_point):
        callback.curves["deterministic"].add(step * 100, result(**rates))
    return callback


@pytest.mark.unit
def test_stops_when_all_stop_skills_reach_the_threshold():
    """Stop only once every stop skill has reached the threshold."""
    callback = callback_with_curve([{"a": 1.0, "b": 0.5}], stop_skills=["a", "b"])
    callback._check_stop()
    assert callback.stop_reason is None
    callback.curves["deterministic"].add(100, result(a=1.0, b=0.96))
    callback._check_stop()
    assert "a, b reached success rate" in callback.stop_reason


@pytest.mark.unit
def test_stops_after_patience_evaluations_without_improvement():
    """With patience 2, two evaluations without a higher mean success rate stop training."""
    callback = callback_with_curve([], patience=2)
    for step, (a, b) in enumerate([(0.2, 0.2), (0.4, 0.2), (0.3, 0.3), (0.2, 0.2)]):
        callback.curves["deterministic"].add(step * 100, result(a=a, b=b))
        callback._check_stop()
        assert (callback.stop_reason is None) == (step < 3)


def callback_with_points(points, **kwargs):
    """A MetricsCallback over skills a and b whose deterministic curve holds the given (step, rates) points."""
    callback = MetricsCallback(None, ["a", "b"], 100, 1, "unused", **kwargs)
    for step, rates in points:
        callback.curves["deterministic"].add(step, result(**rates))
        callback._check_stop()
    return callback


# b is added at step 100 (its zero-shot point) and learned quickly, while a, at 0.95 before, is forgotten and
# relearned: back within 0.1 of its level only from step 500
FORGET_THEN_RECOVER = [
    (0, {"a": 1.0}), (100, {"a": 0.9, "b": 0.0}), (200, {"a": 0.5, "b": 1.0}), (300, {"a": 0.6, "b": 1.0}),
    (400, {"a": 0.9, "b": 1.0}), (500, {"a": 0.9, "b": 0.96}),
]


@pytest.mark.unit
def test_recover_others_waits_until_the_other_skills_are_back_at_their_earlier_level():
    """Control: stopping on the added skill alone stops at step 300, with a forgotten (0.6). With recover_others,
    training goes on until a is within 0.1 of its earlier level (0.95) on 2 evaluations in a row (step 500)."""
    alone = callback_with_points(FORGET_THEN_RECOVER[:4], stop_skills=["b"], stop_consecutive=2)
    assert alone.stop_reason is not None
    for end in range(4, 6):
        retain = callback_with_points(FORGET_THEN_RECOVER[:end], stop_skills=["b"], stop_consecutive=2,
                                      recover_others=True)
        assert retain.stop_reason is None
    retain = callback_with_points(FORGET_THEN_RECOVER, stop_skills=["b"], stop_consecutive=2, recover_others=True)
    assert "every other skill is back" in retain.stop_reason


@pytest.mark.unit
def test_recovery_counts_only_evaluations_after_the_addition():
    """Right after b is added, only b is evaluated (densely). a's evaluations before the addition are at its level,
    so a check over all of a's evaluations would call it recovered (the control); it is not, until a has been
    evaluated again twice."""
    from cam.evaluation.metrics_callback import holds

    points = [(0, {"a": 1.0}), (100, {"a": 1.0, "b": 0.0}), (110, {"b": 1.0}), (120, {"b": 1.0})]
    callback = callback_with_points(points, stop_skills=["b"], stop_consecutive=2, recover_others=True)
    all_of_a = [e.success_rate for _, e in callback.curves["deterministic"].evaluations("a")]
    assert holds(all_of_a, 0.9, 2)
    assert callback.stop_reason is None and callback.retention()["skills"]["a"]["recovered"] is False


@pytest.mark.unit
def test_retention_records_each_skills_level_drop_and_recovery():
    """a: level 0.95 before b was added at step 100, lowest 0.5 after, last 0.9, recovered (2 in a row within
    0.1) at step 500, which is also when all skills were."""
    callback = callback_with_points(FORGET_THEN_RECOVER, stop_skills=["b"], stop_consecutive=2)
    retention = callback.retention()
    assert retention["added_at"] == 100 and set(retention["skills"]) == {"a"}
    assert retention["skills"]["a"] == {
        "level_before": pytest.approx(0.95), "lowest_after": 0.5, "final": 0.9, "recovered_at": 500, "recovered": True,
    }
    assert retention["all_recovered_at"] == 500
    assert callback_with_points(FORGET_THEN_RECOVER[:1]).retention() is None  # nothing added


@pytest.mark.unit
def test_step_returns_false_once_stopping():
    """_on_step tells Stable-Baselines3 to stop (returns False) once a stop reason is set."""
    callback = callback_with_curve([])
    callback.locals = {"dones": [False], "infos": [{"skill": "a"}]}
    callback.next_evaluation_step = 10**9
    assert callback._on_step() is True
    callback.stop_reason = "test"
    assert callback._on_step() is False


@pytest.mark.unit
def test_resume_at_the_last_evaluation_only_evaluates_added_skills(monkeypatch):
    """Resuming at the last evaluated step skips skills already evaluated there and merges added skills
    into that point (no duplicate evaluation at the same step)."""
    import cam.evaluation.metrics_callback as metrics_callback_module
    from cam.evaluation.evaluation import EvaluationResult, SkillEvaluation

    evaluated = []

    def fake_evaluate(env, policy, skill_names, episodes, seed=None):
        evaluated.append(list(skill_names))
        return EvaluationResult({skill: SkillEvaluation(1, 1.0, 0.0, 1.0, 0.0, 5.0, 0.0) for skill in skill_names})

    monkeypatch.setattr(metrics_callback_module, "evaluate", fake_evaluate)
    callback = callback_with_curve([{"a": 0.5, "b": 0.5}])  # one point at step 0
    callback.num_timesteps, callback.model = 0, None
    monkeypatch.setattr(callback, "write_metrics", lambda: None)
    monkeypatch.setattr(type(callback), "logger", property(lambda self: __import__("unittest.mock").mock.MagicMock()))

    callback.skill_names = ["a", "b"]
    callback._on_training_start()
    assert evaluated == []  # nothing new to evaluate at step 0

    callback.skill_names = ["a", "b", "c"]
    callback._on_training_start()
    assert evaluated == [["c"], ["c"]]  # deterministic and stochastic, only the added skill
    assert sorted(callback.curves["deterministic"].points[-1][1].skills) == ["a", "b", "c"]


@pytest.mark.integration
def test_skills_added_on_resume_are_evaluated_densely_then_sparsely(monkeypatch):
    """A skill added on resume is evaluated alone at +10, +20, +40, ... steps (first evaluation 10 here),
    doubling until the regular interval (100) takes over; training stops at the first evaluation at the
    threshold, and no extra evaluation follows the stop."""
    from unittest.mock import MagicMock

    import cam.evaluation.metrics_callback as metrics_callback_module
    from cam.evaluation.evaluation import EvaluationResult, SkillEvaluation

    evaluated = []

    def fake_evaluate(env, policy, skill_names, episodes, seed=None):
        evaluated.append((callback.num_timesteps, list(skill_names)))
        rate = 1.0 if callback.num_timesteps >= 40 else 0.0
        return EvaluationResult({skill: SkillEvaluation(1, rate, 0.0, 1.0, 0.0, 5.0, 0.0) for skill in skill_names})

    monkeypatch.setattr(metrics_callback_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(metrics_callback_module, "ADDED_SKILL_FIRST_EVALUATION", 10)
    callback = MetricsCallback(None, ["a", "b", "c"], 100, 1, "unused", stop_skills=["c"])
    callback.curves["deterministic"].add(0, result(a=0.5, b=0.5))  # trained on a and b; "c" is added on resume
    monkeypatch.setattr(callback, "write_metrics", lambda: None)
    monkeypatch.setattr(type(callback), "logger", property(lambda self: MagicMock()))
    callback.model, callback.num_timesteps = None, 0
    callback.locals = {"dones": [False], "infos": [{"skill": "c"}]}
    callback._on_training_start()
    running = True
    while running:
        callback.num_timesteps += 1
        running = callback._on_step()
    callback._on_training_end()
    assert evaluated[::2] == [(0, ["c"]), (10, ["c"]), (20, ["c"]), (40, ["c"])]  # deterministic of each pair
    assert callback.curves["deterministic"].steps_to_threshold("c", 0.95) == 40


@pytest.mark.unit
def test_a_further_addition_keeps_the_retention_of_the_one_before(tmp_path, monkeypatch):
    """A run that added b at step 100 (FORGET_THEN_RECOVER) is resumed at step 500 with c added: metrics.json keeps b's
    retention, as it was at step 500, in earlier_retention, and "retention" becomes c's. Control: resumed without an
    added skill (after an interruption), nothing moves to earlier_retention."""
    from unittest.mock import MagicMock

    import cam.evaluation.metrics_callback as metrics_callback_module
    from cam.evaluation.evaluation import EvaluationResult, SkillEvaluation
    from cam.evaluation.learning_curve import TrainingProgress

    def fake_evaluate(env, policy, skill_names, episodes, seed=None):
        return EvaluationResult({skill: SkillEvaluation(1, 0.0, 0.0, 1.0, 0.0, 5.0, 0.0) for skill in skill_names})

    monkeypatch.setattr(metrics_callback_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(MetricsCallback, "logger", property(lambda self: MagicMock()))
    first = MetricsCallback(None, ["a", "b"], 100, 1, tmp_path, stop_skills=["b"], stop_consecutive=2)
    for step, rates in FORGET_THEN_RECOVER:
        first.curves["deterministic"].add(step, result(**rates), TrainingProgress(step, 0, {}, {}))
    first.num_timesteps = 500
    first.write_metrics()
    saved = json.loads((tmp_path / "metrics.json").read_text())

    def resume(skills):
        callback = MetricsCallback(None, skills, 100, 1, tmp_path, metrics=saved)
        callback.model, callback.num_timesteps = None, 500
        callback._on_training_start()
        return callback

    assert resume(["a", "b"]).earlier_retention == []
    resume(["a", "b", "c"])
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["retention"]["added_at"] == 100 and metrics["earlier_retention"] == [saved["retention"]]
    assert metrics["retention"]["added_at"] == 500
