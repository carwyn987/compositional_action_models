import itertools

import gymnasium as gym
import numpy as np
import pytest

from cam.environments.fetch.fetch_env_state_annotation_wrapper import FetchEnvStateAnnotationWrapper
from cam.environments.fetch.fetch_multiblock_environment import (
    ENVIRONMENT_ID,
    MAX_BLOCKS,
    MIN_BLOCK_SEPARATION,
)


def make(num_blocks: int) -> FetchEnvStateAnnotationWrapper:
    return FetchEnvStateAnnotationWrapper(gym.make(ENVIRONMENT_ID, num_blocks=num_blocks))


@pytest.mark.integration
@pytest.mark.parametrize("num_blocks", [1, 2, MAX_BLOCKS])
def test_observation_size(num_blocks):
    env = make(num_blocks)
    obs, _ = env.reset(seed=0)
    assert obs["observation"].shape == (25 + 9 * (num_blocks - 1),)
    assert env.observation_space.contains(obs)
    env.close()


@pytest.mark.integration
@pytest.mark.parametrize("seed", range(5))
def test_reset_places_blocks_on_table_apart(seed):
    env = make(MAX_BLOCKS)
    _, info = env.reset(seed=seed)
    positions = list(info["environment_state"].block_positions.values())
    table_z = env.unwrapped.height_offset
    assert all(abs(p[2] - table_z) < 1e-3 for p in positions)
    for a, b in itertools.combinations(positions, 2):
        assert np.linalg.norm(a[:2] - b[:2]) >= MIN_BLOCK_SEPARATION - 1e-6
    env.close()


@pytest.mark.integration
def test_blocks_stay_at_rest_without_action():
    env = make(3)
    _, info = env.reset(seed=0)
    before = info["environment_state"].block_positions
    for _ in range(20):
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
    after = info["environment_state"].block_positions
    for name in before:
        assert np.linalg.norm(after[name] - before[name]) < 1e-3
    env.close()


@pytest.mark.integration
def test_reset_is_deterministic_given_seed():
    first, second = make(3), make(3)
    _, info_a = first.reset(seed=7)
    _, info_b = second.reset(seed=7)
    for name, position in info_a["environment_state"].block_positions.items():
        np.testing.assert_array_equal(position, info_b["environment_state"].block_positions[name])
    first.close()
    second.close()


@pytest.mark.integration
@pytest.mark.parametrize("num_blocks", [0, MAX_BLOCKS + 1])
def test_rejects_invalid_block_count(num_blocks):
    with pytest.raises(ValueError):
        gym.make(ENVIRONMENT_ID, num_blocks=num_blocks)
