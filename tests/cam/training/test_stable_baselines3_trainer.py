import pytest

pytest.importorskip("stable_baselines3")

from main import parse_args  # noqa: E402
from cam.experiments.environment_setup import setup_environment  # noqa: E402
from cam.skills.registry import build_skill  # noqa: E402
from cam.training.stable_baselines3_trainer import train_stable_baselines3  # noqa: E402


@pytest.mark.integration
@pytest.mark.parametrize(
    "algorithm, overrides",
    [
        ("sac", {"learning_starts": 50, "batch_size": 32}),
        ("mtsac", {"learning_starts": 50, "batch_size": 32}),
        ("ppo", {"n_steps": 64, "batch_size": 32}),
    ],
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


def train_pickup(tmp_path, steps, **kwargs):
    config = parse_args(["--skills", "pickup", "--num-blocks", "2", "--max-steps-per-episode", "10"])
    env = setup_environment(config, [build_skill("pickup", config)])
    try:
        return train_stable_baselines3(env, "sac", steps, tmp_path, seed=0,
                                       hyperparameter_overrides={"learning_starts": 50, "batch_size": 32}, **kwargs)
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.parametrize("warmup", [0, 150])
def test_resuming_without_a_replay_buffer_waits_before_updating(tmp_path, warmup):
    """Train 200 steps without saving the buffer, then resume for 200 more. Control (no warm-up): SAC updates from the
    first new step, on a buffer of a handful of transitions. With a 150-step warm-up, only the last 50 steps update."""
    before = train_pickup(tmp_path, 200)._n_updates
    resumed = train_pickup(tmp_path, 200, resume=True, resume_warmup_steps=warmup)
    assert abs(resumed._n_updates - before - (200 - warmup)) <= 1
    assert resumed.gradient_steps == 1


@pytest.mark.integration
def test_a_model_saved_during_a_warm_up_updates_again_when_resumed(tmp_path):
    """A run interrupted during its warm-up saves gradient_steps 0; resuming restores the usual gradient steps, so
    training does not silently stop updating (resumed here with its buffer and no warm-up)."""
    from stable_baselines3 import SAC

    train_pickup(tmp_path, 200, save_replay_buffer=True)
    interrupted = SAC.load(tmp_path / "model.zip")
    interrupted.gradient_steps = 0
    interrupted.save(tmp_path / "model.zip")
    before = interrupted._n_updates
    resumed = train_pickup(tmp_path, 100, resume=True)
    assert resumed.gradient_steps == 1 and resumed._n_updates - before >= 99
