import numpy as np
import pytest

from cam.domain.symbols import Predicate
from cam.environments.fetch.fetch_env_state_annotation_wrapper import FetchState
from cam.environments.fetch.fetch_predicate_evaluation_wrapper import TABLE_REST_Z, FetchPredicateEvaluationWrapper

OPEN, GRIPPING, CLOSED_ON_NOTHING = 0.10, 0.048, 0.0  # finger widths: open, closed on a block, closed
BLOCK1 = np.array([1.4, 0.8, TABLE_REST_Z])  # on the table, away from the gripper and block0
ON_BLOCK1 = BLOCK1 + [0.0, 0.0, 0.05]
FAR = np.array([1.0, 0.5, 0.7])


def facts(gripper, finger_width, block0):
    state = FetchState(np.array(gripper, dtype=float), finger_width, {"block0": np.array(block0), "block1": BLOCK1})
    return FetchPredicateEvaluationWrapper.evaluate(state)


def fact(name, *arguments):
    return Predicate(name, arguments)


@pytest.mark.unit
def test_reset_scene_gripper_closed_away_from_blocks_is_empty():
    """Fetch resets with the fingers closed (width 0): with no block between them the gripper is empty."""
    result = facts(FAR, CLOSED_ON_NOTHING, [1.2, 0.6, TABLE_REST_Z])
    assert {fact("gripper-empty"), fact("on-table", "block0"), fact("clear", "block0")} <= result


@pytest.mark.unit
@pytest.mark.parametrize("support", ["table", "block1"])
def test_gripped_block_is_held_only_once_lifted_off_its_support(support):
    """Closing on a resting block (pickup / unstack before the lift) is neither holding nor gripper-empty."""
    resting = [1.2, 0.6, TABLE_REST_Z] if support == "table" else ON_BLOCK1
    gripped = facts(resting, GRIPPING, resting)
    assert fact("holding", "block0") not in gripped and fact("gripper-empty") not in gripped
    lifted = np.array(resting) + [0.0, 0.0, 0.02]
    assert fact("holding", "block0") in facts(lifted, GRIPPING, lifted)


@pytest.mark.unit
def test_block_placed_on_another_needs_release_for_gripper_empty():
    """stack's effects: on and clear hold once the block rests on the base; gripper-empty only once open."""
    placed_still_gripped = facts(ON_BLOCK1, GRIPPING, ON_BLOCK1)
    assert fact("on", "block0", "block1") in placed_still_gripped
    assert fact("gripper-empty") not in placed_still_gripped
    released = facts(ON_BLOCK1, OPEN, ON_BLOCK1)
    assert {fact("on", "block0", "block1"), fact("clear", "block0"), fact("gripper-empty")} <= released
    assert fact("clear", "block1") not in released and fact("on-table", "block0") not in released


@pytest.mark.unit
def test_held_block_is_not_clear_and_not_on_anything():
    lifted = ON_BLOCK1 + [0.0, 0.0, 0.03]
    result = facts(lifted, GRIPPING, lifted)
    assert fact("holding", "block0") in result
    assert not {fact("clear", "block0"), fact("on", "block0", "block1"), fact("on-table", "block0")} & result
    assert fact("clear", "block1") in result
