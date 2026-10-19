import numpy as np
import pytest

from cam.domain.symbols import Predicate
from cam.environments.fetch.fetch_env_state_annotation_wrapper import FetchState
from cam.environments.fetch.fetch_rewards import (
    FetchMotionReward,
    FetchPickupRaisedReward,
    FetchPickupReward,
    FetchPlaceBesideReward,
    FetchPutdownReward,
    FetchStackReward,
    FetchUnstackReward,
)
from cam.skills.blocksworld_skills import PickupSkill, PutdownSkill, StackSkill, UnstackSkill

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


PUTDOWN = PutdownSkill({"symbolic_action_model_format": "pddl"}).ground({"?o": "block0"})


def putdown_info(height, finger_width=0.048, facts=frozenset(), rotation=(0.0, 0.0, 0.0)):
    block = np.array([1.3, 0.7, 0.425 + height])
    state = FetchState(block, finger_width, {"block0": block}, {"block0": np.array(rotation)})
    return {"environment_state": state, "facts": facts}


@pytest.mark.unit
def test_putdown_penalizes_opening_high_and_dropping_fast():
    reward = FetchPutdownReward()
    reward.reset(PUTDOWN, {})
    reward(PUTDOWN, np.zeros(4), putdown_info(0.10), success=False)
    components = reward(PUTDOWN, np.zeros(4), putdown_info(0.06, finger_width=0.10), success=False)[1]
    assert components["early_open"] < 0 and components["gentle"] < 0  # opened at 6 cm; fell 4 cm in a step


@pytest.mark.unit
def test_putdown_success_bonus_is_larger_for_a_square_placement():
    placed = frozenset({Predicate("on-table", ("block0",)), Predicate("gripper-empty", ())})
    bonuses = []
    for rotation in [(0.0, 0.0, 0.0), (0.0, 0.0, 0.5)]:
        reward = FetchPutdownReward()
        reward.reset(PUTDOWN, {})
        bonuses.append(reward(PUTDOWN, np.zeros(4), putdown_info(0.0, 0.10, placed, rotation), success=True)[1]["success"])
    assert bonuses[0] == pytest.approx(15.0) and bonuses[0] > bonuses[1]


@pytest.mark.unit
def test_pickup_raised_keeps_rewarding_the_lift_past_pickups_target():
    """pickup's lift target is 6 cm; pickup-raised's is the middle of the raised band (33 cm)."""
    def lift_reward(reward_class):
        reward = reward_class()
        reward.reset(PICKUP_BLOCK0, {})
        reward.grip_earned = True
        at = lambda z: {"environment_state": FetchState(np.array([1.3, 0.7, z]), 0.048, {"block0": np.array([1.3, 0.7, z])}), "facts": frozenset()}
        reward(PICKUP_BLOCK0, np.zeros(4), at(0.425 + 0.07), success=False)
        return reward(PICKUP_BLOCK0, np.zeros(4), at(0.425 + 0.09), success=False)[1]["lift"]
    assert lift_reward(FetchPickupReward) == 0 < lift_reward(FetchPickupRaisedReward)


@pytest.mark.unit
def test_place_beside_aligns_to_the_ring_around_the_other_block():
    """Its align target is any point at the beside spacing from the other block, not over it."""
    reward = FetchPlaceBesideReward()
    on_ring = BASE + [reward.SPACING, 0.0, 0.0]
    assert reward.align_distance(on_ring, BASE) == pytest.approx(0.0)
    assert reward.align_distance(BASE, BASE) == pytest.approx(reward.SPACING)
    assert reward.resting_z(on_ring, BASE) == pytest.approx(0.425)


@pytest.mark.unit
def test_motion_reward_penalizes_action_changes_block_acceleration_and_other_blocks_moving():
    """Steady motion costs nothing; changing the action, stopping the moved block dead and moving another
    block each cost."""
    def info(block0, block1):
        return {"environment_state": FetchState(np.zeros(3), 0.05, {"block0": np.array(block0), "block1": np.array(block1)})}

    reward = FetchMotionReward()
    reward.reset(PICKUP_BLOCK0, {})
    steady = [reward(PICKUP_BLOCK0, np.zeros(4), info([0, 0, 0.5 + 0.01 * t], BASE), False)[1] for t in range(3)]
    assert steady[-1] == {"smoothness": 0.0, "acceleration": pytest.approx(0.0), "disturbance": 0.0, "force": 0.0}
    slam = reward(PICKUP_BLOCK0, np.array([0, 0, 1.0, 0]), info([0, 0, 0.52], BASE + [0.01, 0, 0]), False)[1]
    assert slam["smoothness"] < 0 and slam["acceleration"] < 0 and slam["disturbance"] < 0


def force_info(force, earlier=()):
    """A step's info whose net gripper force is force N, pointing down, after the earlier steps' magnitudes."""
    state = FetchState(np.zeros(3), 0.05, {"block0": np.array([1.3, 0.7, 0.425])},
                       gripper_force=np.array([0.0, 0.0, force]), gripper_force_history=(*earlier, force))
    return {"environment_state": state, "facts": frozenset()}


@pytest.mark.unit
def test_gripper_force_is_penalized_up_to_the_failure_limit():
    """Holding a block (20 N) costs nothing; 300 N costs 0.1. A one-step contact spike of 38 kN (seen while
    exploring) costs at most 0.2, where the uncapped penalty (the control) would cost 19, more than success earns
    (15). At most 0.2 per step, a press that lasts until it fails (5 steps) costs a tenth of the failure itself."""
    reward = FetchMotionReward()
    reward.reset(PICKUP_BLOCK0, {})
    assert reward(PICKUP_BLOCK0, np.zeros(4), force_info(20.0), False)[1]["force"] == 0
    from cam.environments.fetch.fetch_predicate_evaluation_wrapper import FORCE_FAILURE_STEPS
    from cam.environments.fetch.fetch_rewards import FAILURE_PENALTY

    assert reward(PICKUP_BLOCK0, np.zeros(4), force_info(300.0), False)[1]["force"] == pytest.approx(-0.1)
    spike = reward(PICKUP_BLOCK0, np.zeros(4), force_info(38_000.0), False)[1]["force"]
    assert spike == pytest.approx(-0.2) == pytest.approx(-FAILURE_PENALTY / FORCE_FAILURE_STEPS / 10)
    assert reward.config.force_penalty * (38_000.0 - 100.0) == pytest.approx(18.95)


@pytest.mark.unit
def test_extreme_gripper_force_fails_only_when_it_lasts_five_steps():
    """Over 500 N for one step (a first-contact spike) or four (a bump) is not a failure; for five steps in a row
    (pressing) it is. Moderate force (300 N) never fails, however long."""
    from cam.environments.fetch.fetch_predicate_evaluation_wrapper import FORCE_FAILURE_STEPS, check_fetch_failure

    start = force_info(0.0)
    assert FORCE_FAILURE_STEPS == 5
    assert check_fetch_failure(PICKUP_BLOCK0, start, force_info(300.0, earlier=[300.0] * 9)) is None
    assert check_fetch_failure(PICKUP_BLOCK0, start, force_info(38_000.0, earlier=[0.0] * 4)) is None
    assert check_fetch_failure(PICKUP_BLOCK0, start, force_info(600.0, earlier=[0.0, 700, 700, 700])) is None
    pressing = check_fetch_failure(PICKUP_BLOCK0, start, force_info(600.0, earlier=[0.0, 700, 700, 700, 700]))
    assert pressing.startswith("excessive force")


def pickup_environment():
    from main import parse_args
    from cam.experiments.environment_setup import setup_environment
    from cam.skills.registry import build_skill

    config = parse_args(["--skills", "pickup", "--num-blocks", "2", "--reward", "shaped", "--seed", "0"])
    return setup_environment(config, [build_skill("pickup", config)])


@pytest.mark.integration
def test_random_exploration_mostly_survives_its_table_bumps():
    """Random actions (how SAC starts exploring) bump the table: over 500 N for a few steps at a time. Recorded over
    20 random pickup episodes, failing on a single step over 500 N (the control) would end most of them; failing
    only after 5 steps in a row ends few."""
    env = pickup_environment()
    env.env.env.check_failure = None  # record every episode to its end
    rng = np.random.default_rng(7)
    longest = []
    try:
        for episode in range(20):
            env.reset(seed=200 + episode)
            run = best = 0
            done = truncated = False
            while not (done or truncated):
                _, _, done, truncated, info = env.step(rng.uniform(-1, 1, 4))
                run = run + 1 if info["environment_state"].gripper_force_history[-1] > 500 else 0
                best = max(best, run)
            longest.append(best)
    finally:
        env.close()
    assert np.mean(np.array(longest) >= 1) >= 0.4
    assert np.mean(np.array(longest) >= 5) <= 0.25


@pytest.mark.integration
def test_pressing_into_the_table_still_fails():
    """Driving the gripper down into the table keeps the force far above 500 N (~55 kN) on every step: the episode
    fails on excessive force, within a few steps of first contact."""
    env = pickup_environment()
    try:
        env.reset(seed=3)
        for step in range(40):
            _, _, done, _, info = env.step(np.array([0.0, 0.0, -1.0, 1.0]))
            if done:
                break
    finally:
        env.close()
    assert done and info["failure"].startswith("excessive force") and step < 15


@pytest.mark.integration
def test_exploring_near_the_block_scores_above_doing_nothing():
    """The scripted pickup with heavy exploration noise (sigma 0.6, as SAC explores early) scores above doing nothing
    with the current weights. Scored on the same episodes with ten times the motion penalties and a fifth of the
    time cost (the control: the weights the 2026-10-14 runs learned nothing with), it scores below: keeping away
    from the blocks paid better than trying."""
    from cam.environments.fetch.fetch_rewards import FetchMotionRewardConfig
    from cam.environments.fetch.fetch_scripted_policies import FetchScriptedPickupPolicy

    heavy = FetchMotionReward(FetchMotionRewardConfig(
        smoothness_penalty=0.005, acceleration_penalty=500.0, disturbance_penalty=20.0, force_penalty=0.005))
    motion = ("smoothness", "acceleration", "disturbance", "force")
    env = pickup_environment()
    rng = np.random.default_rng(0)
    try:
        returns = {"current": [], "control": [], "idle": []}
        for episode in range(10):
            for noisy in (True, False):
                policy = FetchScriptedPickupPolicy()
                policy.reset()
                obs, info = env.reset(seed=1000 + episode)
                heavy.reset(info["grounded_action_model"], info)
                current = control = 0.0
                done = truncated = False
                while not (done or truncated):
                    action = np.zeros(4)
                    if noisy:
                        action = np.clip(policy(obs, info, info["grounded_action_model"]) + rng.normal(0, 0.6, 4), -1, 1)
                    obs, reward, done, truncated, info = env.step(action)
                    components = info["reward_components"]
                    current += reward
                    heavy_terms = heavy(info["grounded_action_model"], action, info, False)[1]
                    control += (reward - sum(components[name] for name in motion) - 0.8 * components["time"]
                                + sum(heavy_terms.values()))
                if noisy:
                    returns["current"].append(current)
                    returns["control"].append(control)
                else:
                    returns["idle"].append(current)
    finally:
        env.close()
    idle_control = np.mean(returns["idle"]) + 0.8 * 0.05 * 100  # doing nothing costs only the time, a fifth of it
    assert np.mean(returns["current"]) > np.mean(returns["idle"])
    assert np.mean(returns["control"]) < idle_control
