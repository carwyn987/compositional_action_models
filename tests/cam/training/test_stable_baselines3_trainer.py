import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3.common.buffers import DictReplayBuffer  # noqa: E402

from main import parse_args  # noqa: E402
from cam.experiments.environment_setup import setup_environment  # noqa: E402
from cam.skills.registry import build_skill  # noqa: E402
from cam.training.stable_baselines3_trainer import remove_transitions, train_stable_baselines3  # noqa: E402


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


@pytest.mark.integration
def test_training_writes_its_progress_for_the_overnight_runner(tmp_path):
    """A 300-step run ends with progress.json at 300 of 300 steps; resumed for 200 more, it counts only those."""
    import json

    train_pickup(tmp_path, 300)
    first = json.loads((tmp_path / "progress.json").read_text())
    train_pickup(tmp_path, 200, resume=True, resume_warmup_steps=0)
    resumed = json.loads((tmp_path / "progress.json").read_text())
    assert first == {"done": 300, "total": 300, "step": 300}
    assert resumed == {"done": 200, "total": 200, "step": 500}


KEPT, REMOVED = np.eye(2, dtype=np.float32)


def replay_buffer(skills: str, size: int) -> DictReplayBuffer:
    """A replay buffer of size slots given one transition per letter of skills (K: the kept skill's embedding, R: the
    removed one's). Transition i has observation i, next observation i + 0.5, action i / 100, reward -i, done on
    every third and a timeout on every fourth, so each array shows whether it moved with its transition."""
    buffer = DictReplayBuffer(size, gym.spaces.Dict({
        "observation": gym.spaces.Box(-np.inf, np.inf, (1,)), "operator_embedding": gym.spaces.Box(0, 1, (2,)),
    }), gym.spaces.Box(-1, 1, (1,)), device="cpu")
    for i, skill in enumerate(skills):
        embedding = (KEPT if skill == "K" else REMOVED)[None]
        buffer.add({"observation": np.array([[i]], np.float32), "operator_embedding": embedding},
                   {"observation": np.array([[i + 0.5]], np.float32), "operator_embedding": embedding},
                   np.array([[i / 100]], np.float32), np.array([-i], np.float32), np.array([i % 3 == 0]),
                   [{"TimeLimit.truncated": i % 4 == 0}])
    return buffer


def assert_holds(buffer: DictReplayBuffer, transitions: list[int]) -> None:
    """The buffer holds exactly these transitions (by index), in this order from its first slot, each whole."""
    count = len(transitions)
    i = np.array(transitions, dtype=np.float32)
    assert buffer.size() == count
    np.testing.assert_array_equal(buffer.observations["observation"][:count, 0, 0], i)
    np.testing.assert_array_equal(buffer.observations["operator_embedding"][:count, 0], np.tile(KEPT, (count, 1)))
    np.testing.assert_array_equal(buffer.next_observations["observation"][:count, 0, 0], i + 0.5)
    np.testing.assert_array_equal(buffer.next_observations["operator_embedding"][:count, 0], np.tile(KEPT, (count, 1)))
    np.testing.assert_array_equal(buffer.actions[:count, 0, 0], i / 100)
    np.testing.assert_array_equal(buffer.rewards[:count, 0], -i)
    np.testing.assert_array_equal(buffer.dones[:count, 0], i % 3 == 0)
    np.testing.assert_array_equal(buffer.timeouts[:count, 0], i % 4 == 0)


def sampled(buffer: DictReplayBuffer) -> set[float]:
    """The transitions (observations) 500 samples draw; asserts each sample is one whole transition."""
    batch = buffer.sample(500)
    observations = batch.observations["observation"][:, 0].numpy()
    np.testing.assert_array_equal(batch.next_observations["observation"][:, 0].numpy(), observations + 0.5)
    np.testing.assert_allclose(batch.actions[:, 0].numpy(), observations / 100)
    np.testing.assert_array_equal(batch.rewards[:, 0].numpy(), -observations)
    return set(observations.tolist())


@pytest.mark.unit
def test_removing_a_skills_transitions_keeps_the_others_whole_and_in_order():
    """Six transitions in a buffer of ten, alternating two skills: removing one skill's leaves the other's three at
    the front, each with its own next observation, action, reward, done and timeout, and sampling draws only them.
    Control: before the removal, sampling draws both skills' transitions."""
    buffer = replay_buffer("KRKRKR", 10)
    assert sampled(buffer) == {0, 1, 2, 3, 4, 5}
    assert remove_transitions(buffer, [REMOVED]) == 3
    assert (buffer.pos, buffer.full) == (3, False)
    assert_holds(buffer, [0, 2, 4])
    assert sampled(buffer) == {0, 2, 4}


@pytest.mark.unit
def test_removing_transitions_from_a_full_wrapped_buffer_keeps_the_oldest_first():
    """14 transitions in a buffer of ten: it is full, and its oldest (transition 4) is at slot 4, after the newest.
    Removing one skill's transitions leaves the other's from oldest to newest at the front, the buffer no longer full,
    so the next transitions fill the freed slots and then replace the oldest kept one first. Control: a skill with no
    transitions in the buffer removes nothing and leaves it as it was."""
    skills = "KRRKRKKRKRRKKR"
    buffer = replay_buffer(skills, 10)
    assert (buffer.pos, buffer.full) == (4, True)
    assert remove_transitions(buffer, [np.array([0.5, 0.5], np.float32)]) == 0
    assert (buffer.pos, buffer.full) == (4, True)
    assert sampled(buffer) == set(range(4, 14))

    kept = [i for i in range(4, 14) if skills[i] == "K"]  # 5, 6, 8, 11, 12
    assert remove_transitions(buffer, [REMOVED]) == 10 - len(kept)
    assert (buffer.pos, buffer.full) == (len(kept), False)
    assert_holds(buffer, kept)
    assert sampled(buffer) == set(kept)

    for i in range(100, 100 + 10 - len(kept) + 1):  # fills the buffer, then one more
        buffer.add({"observation": np.array([[i]], np.float32), "operator_embedding": KEPT[None]},
                   {"observation": np.array([[i + 0.5]], np.float32), "operator_embedding": KEPT[None]},
                   np.array([[i / 100]], np.float32), np.array([-i], np.float32), np.array([False]), [{}])
    assert buffer.full and buffer.observations["observation"][0, 0, 0] == 100 + 10 - len(kept)
    assert sampled(buffer) == set(kept[1:]) | set(range(100, 100 + 10 - len(kept) + 1))
