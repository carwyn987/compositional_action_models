import gymnasium as gym
import numpy as np
import pytest

import cam.environments.fetch  # noqa: F401  (registers Fetch environments)
from cam.environments.fetch.fetch_env_state_annotation_wrapper import FetchEnvStateAnnotationWrapper, FetchState


@pytest.mark.unit
def test_extract_state_reads_observation_layout():
    raw = np.arange(25, dtype=np.float64)
    state = FetchEnvStateAnnotationWrapper.extract_state({"observation": raw})
    np.testing.assert_array_equal(state.gripper_position, [0, 1, 2])
    np.testing.assert_array_equal(state.block_positions["block0"], [3, 4, 5])
    assert state.finger_width == 9 + 10


@pytest.mark.unit
def test_extract_state_reads_extra_blocks():
    raw = np.arange(25 + 9 * 2, dtype=np.float64)
    state = FetchEnvStateAnnotationWrapper.extract_state({"observation": raw})
    assert list(state.block_positions) == ["block0", "block1", "block2"]
    np.testing.assert_array_equal(state.block_positions["block1"], [25, 26, 27])
    np.testing.assert_array_equal(state.block_positions["block2"], [34, 35, 36])


@pytest.mark.unit
@pytest.mark.parametrize("size, blocks", [(25, 1), (34, 2), (61, 5)])
def test_count_blocks(size, blocks):
    assert FetchEnvStateAnnotationWrapper.count_blocks(size) == blocks


@pytest.mark.unit
@pytest.mark.parametrize("size", [10, 24, 26, 33])
def test_count_blocks_rejects_other_layouts(size):
    with pytest.raises(ValueError):
        FetchEnvStateAnnotationWrapper.count_blocks(size)


@pytest.mark.unit
def test_extract_state_copies_arrays():
    raw = np.zeros(25)
    state = FetchEnvStateAnnotationWrapper.extract_state({"observation": raw})
    raw[:] = 1.0
    assert state.gripper_position.sum() == 0
    assert state.block_positions["block0"].sum() == 0


@pytest.fixture
def pick_and_place():
    env = FetchEnvStateAnnotationWrapper(gym.make("FetchPickAndPlace-v4"))
    yield env
    env.close()


@pytest.mark.integration
def test_reset_and_step_annotate_environment_state(pick_and_place):
    obs, info = pick_and_place.reset(seed=0)
    assert isinstance(info["environment_state"], FetchState)
    np.testing.assert_array_equal(info["environment_state"].block_positions["block0"], obs["observation"][3:6])

    obs, _, _, _, info = pick_and_place.step(pick_and_place.action_space.sample())
    np.testing.assert_array_equal(info["environment_state"].gripper_position, obs["observation"][0:3])


@pytest.mark.integration
def test_rejects_fetch_task_without_block():
    with pytest.raises(ValueError):
        FetchEnvStateAnnotationWrapper(gym.make("FetchReach-v4"))


@pytest.mark.integration
@pytest.mark.parametrize("num_blocks", [1, 3])
def test_wrapper_annotates_multiblock_environment(num_blocks):
    env = FetchEnvStateAnnotationWrapper(gym.make("FetchMultiBlock-v0", num_blocks=num_blocks))
    obs, info = env.reset(seed=0)
    assert env.num_blocks == num_blocks
    assert list(info["environment_state"].block_positions) == [f"block{i}" for i in range(num_blocks)]
    assert info["objects"] == {f"block{i}": "block" for i in range(num_blocks)}
    env.close()


@pytest.mark.integration
def test_goal_keys_are_dropped(pick_and_place):
    """Observations keep only "observation"; Fetch's goal keys are removed from the space and the data."""
    obs, _ = pick_and_place.reset(seed=0)
    assert list(obs) == ["observation"]
    assert pick_and_place.observation_space.contains(obs)
    obs, *_ = pick_and_place.step(pick_and_place.action_space.sample())
    assert list(obs) == ["observation"]


@pytest.mark.integration
def test_object_features_are_position_and_offset_from_gripper(pick_and_place):
    """info["object_features"][block] = block position followed by block position minus gripper position."""
    _, info = pick_and_place.reset(seed=0)
    state = info["environment_state"]
    features = info["object_features"]["block0"]
    np.testing.assert_allclose(features[:3], state.block_positions["block0"], rtol=1e-6)
    np.testing.assert_allclose(features[3:], state.block_positions["block0"] - state.gripper_position, rtol=1e-5)


@pytest.mark.integration
def test_gripper_force_is_the_carried_blocks_weight_while_holding_it():
    """The net contact force on the gripper while it holds a 2 kg block up is about its weight (~20 N, pointing
    down on the gripper), and zero with nothing touched; the grasp's squeeze cancels between the fingers."""
    from main import parse_args
    from cam.environments.fetch.fetch_scripted_policies import FetchScriptedPickupPolicy
    from cam.experiments.environment_setup import setup_environment
    from cam.skills.registry import build_skill

    config = dict(parse_args(["--skills", "pickup", "--num-blocks", "2"]))
    env = setup_environment(config, [build_skill("pickup", config)])
    obs, info = env.reset(seed=0)
    assert np.linalg.norm(info["environment_state"].gripper_force) == 0
    policy = FetchScriptedPickupPolicy()
    policy.reset()
    for _ in range(40):  # past the success, lifting toward the policy's target
        obs, *_, info = env.step(policy(obs, info, info["grounded_action_model"]))
    force = info["environment_state"].gripper_force
    env.close()
    assert 15 < np.linalg.norm(force) < 30 and force[2] < 0


@pytest.mark.integration
def test_gripper_force_history_restarts_at_each_reset_and_keeps_the_last_steps():
    """The force history holds this episode's steps only (the reset observation first), at most
    GRIPPER_FORCE_HISTORY of them, newest last."""
    from main import parse_args
    from cam.environments.fetch.fetch_env_state_annotation_wrapper import FetchEnvStateAnnotationWrapper
    from cam.experiments.environment_setup import setup_environment
    from cam.skills.registry import build_skill

    config = dict(parse_args(["--skills", "pickup", "--num-blocks", "2"]))
    env = setup_environment(config, [build_skill("pickup", config)])
    try:
        _, info = env.reset(seed=0)
        assert len(info["environment_state"].gripper_force_history) == 1
        for _ in range(15):
            _, *_, info = env.step(np.zeros(4))
        history = info["environment_state"].gripper_force_history
        assert len(history) == FetchEnvStateAnnotationWrapper.GRIPPER_FORCE_HISTORY
        assert history[-1] == pytest.approx(float(np.linalg.norm(info["environment_state"].gripper_force)))
        _, info = env.reset(seed=1)
        assert len(info["environment_state"].gripper_force_history) == 1
    finally:
        env.close()
