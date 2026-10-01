import numpy as np
import pytest

from cam.domain.symbols import Predicate
from cam.environments.fetch.fetch_env_state_annotation_wrapper import FetchState
from cam.environments.fetch.fetch_rewards import FetchPickupReward
from cam.skills.pickup_skill import PickupSkill

PICKUP_BLOCK0 = PickupSkill({"symbolic_action_model_format": "pddl"}).ground({"?o": "block0"})
BLOCK0 = np.array([1.3, 0.7, 0.425])


def info_with_gripper_at(position, facts=frozenset()):
    state = FetchState(np.array(position), 0.08, {"block0": BLOCK0})
    return {"environment_state": state, "facts": facts}


@pytest.mark.unit
def test_moving_toward_the_block_earns_reach_reward():
    """Reducing the gripper-to-block distance gives positive reach reward; moving away gives negative."""
    reward = FetchPickupReward()
    reward.reset(PICKUP_BLOCK0, {})
    reward(PICKUP_BLOCK0, np.zeros(4), info_with_gripper_at([1.0, 1.0, 0.6]), success=False)
    _, closer = reward(PICKUP_BLOCK0, np.zeros(4), info_with_gripper_at([1.1, 0.9, 0.6]), success=False)
    _, farther = reward(PICKUP_BLOCK0, np.zeros(4), info_with_gripper_at([1.0, 1.0, 0.6]), success=False)
    assert closer["reach"] > 0 > farther["reach"]


@pytest.mark.unit
def test_success_bonus_is_paid_once_per_episode():
    """The success bonus appears on the first successful step only, and again after reset."""
    reward = FetchPickupReward()
    reward.reset(PICKUP_BLOCK0, {})
    held = info_with_gripper_at(BLOCK0, frozenset({Predicate("holding", ("block0",))}))
    assert reward(PICKUP_BLOCK0, np.zeros(4), held, success=True)[1]["success"] > 0
    assert reward(PICKUP_BLOCK0, np.zeros(4), held, success=True)[1]["success"] == 0
    reward.reset(PICKUP_BLOCK0, {})
    assert reward(PICKUP_BLOCK0, np.zeros(4), held, success=True)[1]["success"] > 0
