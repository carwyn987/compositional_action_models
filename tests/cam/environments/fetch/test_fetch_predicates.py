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


@pytest.mark.unit
def test_raised_needs_holding_high_above_the_table():
    """pickup succeeds 1 cm up; raised needs the held block 30 cm up (25 cm above a stacked start)."""
    low = [1.2, 0.6, TABLE_REST_Z + 0.21]
    high = [1.2, 0.6, TABLE_REST_Z + 0.31]
    assert fact("raised", "block0") not in facts(low, GRIPPING, low)
    assert {fact("holding", "block0"), fact("raised", "block0")} <= facts(high, GRIPPING, high)
    assert fact("raised", "block0") not in facts(FAR, OPEN, high)  # not held (falling)
    too_high = [1.2, 0.6, TABLE_REST_Z + 0.40]
    assert fact("raised", "block0") not in facts(too_high, GRIPPING, too_high)  # above the band


@pytest.mark.unit
@pytest.mark.parametrize("distance, expected", [(0.075, True), (0.05, False), (0.12, False)])
def test_beside_means_side_by_side_on_the_table(distance, expected):
    block0 = BLOCK1 + [distance, 0.0, 0.0]
    result = facts(FAR, OPEN, block0)
    assert (fact("beside", "block0", "block1") in result) is expected
    assert (fact("beside", "block1", "block0") in result) is expected
    lifted = block0 + [0.0, 0.0, 0.05]
    assert fact("beside", "block0", "block1") not in facts(lifted, GRIPPING, lifted)


@pytest.mark.unit
def test_dropped_means_let_go_in_the_air():
    from cam.environments.fetch.fetch_predicate_evaluation_wrapper import dropped
    from cam.skills.blocksworld_skills import PutdownSkill

    putdown = PutdownSkill({"symbolic_action_model_format": "pddl"}).ground({"?o": "block0"})

    def info(gripper, finger_width, block0):
        state = FetchState(np.array(gripper), finger_width, {"block0": np.array(block0), "block1": BLOCK1})
        return {"environment_state": state, "facts": FetchPredicateEvaluationWrapper.evaluate(state)}

    high = [1.2, 0.6, TABLE_REST_Z + 0.10]
    assert dropped(putdown, {}, info(FAR, OPEN, high))  # let go 10 cm up
    assert not dropped(putdown, {}, info(high, GRIPPING, high))  # still held
    assert not dropped(putdown, {}, info(FAR, OPEN, [1.2, 0.6, TABLE_REST_Z + 0.01]))  # released just above the table
    assert not dropped(putdown, {}, info(FAR, OPEN, ON_BLOCK1))  # resting on a block
    assert not dropped(putdown, {}, info(FAR, OPEN, ON_BLOCK1 + [0, 0, 0.015]))  # released 1.5 cm above a block
    assert dropped(putdown, {}, info(FAR, OPEN, ON_BLOCK1 + [0, 0, 0.05]))  # let go 5 cm above a block


@pytest.mark.unit
def test_regions_are_global_areas_of_the_table_and_push_goals_lie_inside_them():
    from cam.environments.fetch.fetch_predicate_evaluation_wrapper import REGION_CENTRE, push_goal
    from cam.skills.registry import build_skill

    centre = np.array([*REGION_CENTRE, TABLE_REST_Z])
    left_front = centre + [-0.12, 0.12, 0.0]
    result = facts(FAR, OPEN, left_front)
    assert {fact("at-left", "block0"), fact("at-front", "block0")} <= result
    assert not {fact("at-right", "block0"), fact("at-back", "block0")} & result
    assert fact("at-left", "block0") not in facts(left_front + [0, 0, 0.05], GRIPPING, left_front + [0, 0, 0.05])  # held
    assert fact("at-left", "block0") not in facts(FAR, OPEN, centre + [0.0, 0.20, 0.0])  # past the band

    state = FetchState(np.array(FAR), OPEN, {"block0": centre.copy(), "block1": BLOCK1})
    diagonal = build_skill("push-left-front", {"symbolic_action_model_format": "pddl"}).ground({"?o": "block0"})
    goal = push_goal(state, diagonal)
    goal_state = FetchState(np.array(FAR), OPEN, {"block0": np.array([*goal, TABLE_REST_Z]), "block1": BLOCK1})
    assert diagonal.effects_hold(FetchPredicateEvaluationWrapper.evaluate(goal_state))


@pytest.mark.unit
def test_disturbed_names_another_block_moved_from_its_start():
    from cam.environments.fetch.fetch_predicate_evaluation_wrapper import disturbed, check_fetch_failure
    from cam.skills.blocksworld_skills import PickupSkill

    pickup = PickupSkill({"symbolic_action_model_format": "pddl"}).ground({"?o": "block0"})

    def info(block0, block1):
        state = FetchState(np.array(FAR), OPEN, {"block0": np.array(block0), "block1": np.array(block1)})
        return {"environment_state": state, "facts": FetchPredicateEvaluationWrapper.evaluate(state)}

    start = info([1.2, 0.6, TABLE_REST_Z], BLOCK1)
    moved_block0 = info([1.2, 0.6, TABLE_REST_Z + 0.2], BLOCK1)  # the moved block may go anywhere
    nudged = info([1.2, 0.6, TABLE_REST_Z], BLOCK1 + [0.02, 0, 0])
    knocked = info([1.2, 0.6, TABLE_REST_Z], BLOCK1 + [0.06, 0, 0])
    assert disturbed(pickup, start, moved_block0) is None and disturbed(pickup, start, nudged) is None
    assert disturbed(pickup, start, knocked) == "block1"
    assert check_fetch_failure(pickup, start, knocked) == "disturbed block1"


@pytest.mark.unit
def test_a_push_path_is_clear_unless_another_block_lies_between_behind_the_block_and_its_goal():
    from cam.environments.fetch.fetch_predicate_evaluation_wrapper import REGION_CENTRE, push_path_clear
    from cam.skills.registry import build_skill

    def clear(skill, block1, arguments=("block0",)):
        start = np.array([*REGION_CENTRE, TABLE_REST_Z])
        state = FetchState(np.array(FAR), OPEN, {"block0": start, "block1": np.array([*block1, TABLE_REST_Z])})
        grounded = build_skill(skill, {"symbolic_action_model_format": "pddl"}).ground(
            dict(zip(("?o", "?u"), arguments)))
        info = {"environment_state": state, "facts": FetchPredicateEvaluationWrapper.evaluate(state)}
        return push_path_clear(grounded, info)

    x, y = REGION_CENTRE
    assert clear("push-left", (x, y + 0.23))  # past the goal (y + 0.14) by more than the clearance
    assert clear("push-left", (x + 0.10, y + 0.07))  # beside the path
    assert not clear("push-left", (x, y + 0.07))  # on the path
    assert not clear("push-left", (x + 0.05, y + 0.07))  # touching the path
    assert not clear("push-left", (x, y - 0.10))  # where the gripper comes down behind the block
    assert not clear("push-left-front", (x - 0.07, y + 0.07))  # between the legs of the diagonal
    assert clear("pickup", (x, y + 0.07))  # not a push: the block is lifted over the others
    assert clear("push-beside", (x, y + 0.15), ("block0", "block1"))  # ?u, at the goal's end of the path
