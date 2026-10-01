import pytest

pytest.importorskip("stable_baselines3")

from main import parse_args, setup_environment  # noqa: E402
from cam.skills.registry import build_skill  # noqa: E402
from cam.training.stable_baselines3_trainer import train_stable_baselines3  # noqa: E402


@pytest.mark.integration
@pytest.mark.parametrize(
    "algorithm, overrides",
    [("sac", {"learning_starts": 50, "batch_size": 32}), ("ppo", {"n_steps": 64, "batch_size": 32})],
)
def test_short_training_run_saves_model_and_monitor(tmp_path, algorithm, overrides):
    """A few hundred steps on pickup + putdown train end to end and write model.zip and monitor.csv."""
    config = parse_args(["--skills", "pickup", "putdown", "--num-blocks", "2", "--max-steps-per-episode", "20"])
    skills = [build_skill(name, config) for name in config["skills"]]
    env = setup_environment(config, skills)
    try:
        train_stable_baselines3(env, algorithm, 200, tmp_path, seed=0, hyperparameter_overrides=overrides)
    finally:
        env.close()
    assert (tmp_path / "model.zip").exists()
    assert (tmp_path / "monitor.csv").exists()
