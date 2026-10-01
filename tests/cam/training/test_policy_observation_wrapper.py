import gymnasium as gym
import numpy as np
import pytest

from cam.domain.pddl.pddl_parser import parse_operator
from cam.representations.grounding_encoder import GroundingEncoder
from cam.representations.one_hot_operator_encoder import OneHotOperatorEncoder
from cam.training.policy_observation_wrapper import PolicyObservationWrapper
from tests.cam.domain.conftest import LEGACY_OPERATORS

PICKUP = parse_operator(LEGACY_OPERATORS["pickup"])


class OneEpisodeEnv(gym.Env):
    """Provides the info keys the wrapper reads: a pickup(block1) episode over two blocks."""

    observation_space = gym.spaces.Dict({"observation": gym.spaces.Box(-1, 1, (2,))})
    action_space = gym.spaces.Box(-1, 1, (1,))

    def reset(self, *, seed=None, options=None):
        return {"observation": np.zeros(2, dtype=np.float32)}, self._info()

    def step(self, action):
        return {"observation": np.zeros(2, dtype=np.float32)}, 0.0, False, False, self._info()

    def _info(self):
        return {
            "grounded_action_model": PICKUP.ground({"?o": "block1"}),
            "objects": {"block0": "block", "block1": "block"},
            "object_features": {"block0": np.zeros(6), "block1": np.ones(6)},
        }


@pytest.mark.unit
def test_policy_observation_combines_state_operator_and_grounding():
    """Observations hold the state, the lifted operator's embedding, and the grounding slots, within the space."""
    env = PolicyObservationWrapper(OneEpisodeEnv(), OneHotOperatorEncoder([PICKUP]), GroundingEncoder(["block"], 2, 6, 1))
    for obs in (env.reset()[0], env.step(np.zeros(1))[0]):
        assert env.observation_space.contains(obs)
        np.testing.assert_array_equal(obs["operator_embedding"], [1])
        np.testing.assert_array_equal(obs["grounding"], [1, 1, 0, 1, 1, 1, 1, 1, 1, 1])
