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
