import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from main import parse_args  # noqa: E402
from cam.experiments.environment_setup import setup_environment  # noqa: E402
from cam.skills.registry import build_skill  # noqa: E402
from cam.training.checkpointing import atomic_write_text  # noqa: E402
from cam.training.stable_baselines3_trainer import train_stable_baselines3  # noqa: E402

SAC_OVERRIDES = {"learning_starts": 50, "batch_size": 32}


def make_env():
    config = parse_args(["--skills", "pickup", "--num-blocks", "2", "--max-steps-per-episode", "10"])
    return setup_environment(config, [build_skill("pickup", config)])


class InterruptAt(BaseCallback):
    """Raises KeyboardInterrupt at a given step, as Ctrl+C would."""

    def __init__(self, step):
        super().__init__()
        self.step = step

    def _on_step(self):
        if self.num_timesteps >= self.step:
            raise KeyboardInterrupt
        return True


@pytest.mark.unit
def test_atomic_write_replaces_the_file_and_leaves_no_partial(tmp_path):
    """The target holds the new text and no .partial file remains."""
    path = tmp_path / "metrics.json"
    atomic_write_text(path, "old")
    atomic_write_text(path, "new")
    assert path.read_text() == "new"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.integration
def test_periodic_save_writes_a_loadable_model(tmp_path):
    """With save_interval 100, model.zip exists after training, loads, and no partial file is left."""
    env = make_env()
    try:
        train_stable_baselines3(env, "sac", 150, tmp_path, hyperparameter_overrides=SAC_OVERRIDES, save_interval=100)
    finally:
        env.close()
    assert SAC.load(tmp_path / "model.zip").num_timesteps == 150
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.integration
def test_interrupted_training_still_saves_the_model(tmp_path):
    """A Ctrl+C mid-training propagates, but model.zip is saved first at the interrupted step."""
    env = make_env()
    try:
        with pytest.raises(KeyboardInterrupt):
            train_stable_baselines3(
                env, "sac", 1000, tmp_path, hyperparameter_overrides=SAC_OVERRIDES, callbacks=[InterruptAt(120)]
            )
    finally:
        env.close()
    assert SAC.load(tmp_path / "model.zip").num_timesteps == 120
