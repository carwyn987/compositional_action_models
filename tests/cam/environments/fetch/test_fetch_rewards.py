import numpy as np
import pytest

from cam.domain.symbols import Predicate
from cam.environments.fetch.fetch_env_state_annotation_wrapper import FetchState
from cam.environments.fetch.fetch_rewards import FetchPickupReward, FetchStackReward, FetchUnstackReward
from cam.skills.pickup_skill import PickupSkill
from cam.skills.stack_skill import StackSkill
from cam.skills.unstack_skill import UnstackSkill

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


STACK = StackSkill({"symbolic_action_model_format": "pddl"}).ground({"?o": "block0", "?u": "block1"})
BASE = np.array([1.4, 0.8, 0.425])


def stack_info(block, finger_width=0.048, facts=frozenset()):
    state = FetchState(np.array(block), finger_width, {"block0": np.array(block), "block1": BASE})
    return {"environment_state": state, "facts": facts}


@pytest.mark.unit
def test_stack_rewards_raising_the_held_block_and_not_a_dropped_one():
    held = frozenset({Predicate("holding", ("block0",))})
    reward = FetchStackReward()
    reward.reset(STACK, {})
    reward(STACK, np.zeros(4), stack_info([1.2, 0.6, 0.45], facts=held), success=False)
    assert reward(STACK, np.zeros(4), stack_info([1.2, 0.6, 0.47], facts=held), success=False)[1]["carry"] > 0
    reward.reset(STACK, {})
    reward(STACK, np.zeros(4), stack_info([1.2, 0.6, 0.45]), success=False)
    assert reward(STACK, np.zeros(4), stack_info([1.2, 0.6, 0.47]), success=False)[1]["carry"] == 0


@pytest.mark.unit
def test_stack_stage_bonuses_are_paid_once_in_order():
    """carried -> aligned -> placed -> released -> success, each bonus once."""
    held = frozenset({Predicate("holding", ("block0",))})
    on = Predicate("on", ("block0", "block1"))
    steps = [
        (stack_info([1.2, 0.6, 0.53], facts=held), "carry"),  # bottom 5 cm above the base's top
        (stack_info([1.4, 0.8, 0.53], facts=held), "align"),
        (stack_info([1.4, 0.8, 0.475], facts=frozenset({on})), "placed"),
        (stack_info([1.4, 0.8, 0.475], 0.10, frozenset({on, Predicate("gripper-empty", ())})), "release"),
    ]
    reward = FetchStackReward()
    reward.reset(STACK, {})
    for info, stage in steps:
        assert reward(STACK, np.zeros(4), info, success=False)[1][stage] > 1.0
    final = steps[-1][0]
    assert reward(STACK, np.zeros(4), final, success=True)[1]["success"] > 0
    repeated = reward(STACK, np.zeros(4), final, success=True)[1]
    assert repeated["success"] == repeated["placed"] == 0 and repeated["release"] < 1.0


@pytest.mark.unit
def test_unstack_measures_the_lift_from_the_block_below():
    """A block resting on another is already 5 cm up: unstack's lift starts from there, not the table."""
    unstack = UnstackSkill({"symbolic_action_model_format": "pddl"}).ground({"?o": "block0", "?u": "block1"})
    on_base = BASE + [0.0, 0.0, 0.05]
    reset_info = {"environment_state": FetchState(on_base, 0.0, {"block0": on_base, "block1": BASE})}
    reward = FetchUnstackReward()
    reward.reset(unstack, reset_info)
    resting = {**reset_info, "facts": frozenset()}
    assert reward(unstack, np.zeros(4), resting, success=False)[1]["lift_off"] == 0
    lifted = on_base + [0.0, 0.0, 0.02]
    lifted_info = {"environment_state": FetchState(lifted, 0.048, {"block0": lifted, "block1": BASE}), "facts": frozenset()}
    assert reward(unstack, np.zeros(4), lifted_info, success=False)[1]["lift_off"] > 0
