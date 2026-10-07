import json

import pytest

pytest.importorskip("stable_baselines3")

from main import parse_args  # noqa: E402
from cam.evaluation.metrics_callback import MetricsCallback  # noqa: E402
from cam.experiments.environment_setup import setup_environment  # noqa: E402
from cam.skills.registry import build_skill  # noqa: E402
from cam.training.stable_baselines3_trainer import train_stable_baselines3  # noqa: E402


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
