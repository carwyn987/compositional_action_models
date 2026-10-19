"""Wrapper that evaluates symbolic predicates on the annotated Fetch state.

FETCH_PREDICATES maps each predicate name (as written in the symbolic action
models) to a function `(state, *objects) -> bool`. The number of object
parameters after `state` is the predicate's arity. evaluate() grounds every
predicate over every ordered tuple of distinct blocks in the scene, so the same
definitions serve any number of blocks.
"""

import inspect
import logging
from itertools import permutations
from typing import Callable

import gymnasium as gym
import numpy as np

from cam.domain.symbols import Predicate, format_facts
from cam.environments.fetch.fetch_env_state_annotation_wrapper import FetchState
from cam.environments.fetch.fetch_multiblock_environment import BLOCK_HALF_SIZE

logger = logging.getLogger(__name__)

TABLE_REST_Z = 0.425  # block centre height when resting on the table
Z_TOLERANCE = 0.01  # height tolerance for resting on the table / on another block
GRASP_DISTANCE = 0.03  # max gripper-to-block-centre distance for gripping
OPEN_FINGER_WIDTH = 0.07  # finger width at or above which the gripper grips nothing
# Height band above table rest (min, max) in which a held block counts as raised. Well above what pickup or
# unstack reach on their own (they succeed ~1 cm off their support and must stay below it: (not (raised ?o))),
# 25 cm from a stacked start (at least 5 full-speed control steps of at most 5 cm), and bounded above, so the
# block has to be stopped in the band and held there for --success-hold-steps: neither one upward burst nor
# lifting on and on satisfies it. The scripted lift and the raised rewards aim at the middle of the band.
RAISED_HEIGHT = (0.30, 0.36)
BESIDE_DISTANCE = (0.055, 0.10)  # centre-to-centre distance range for two blocks side by side on the table
BESIDE_SPACING = float(np.mean(BESIDE_DISTANCE))  # the middle of that range, where placing / pushing aims

# A block is *supported* when it rests on the table or on another block (on-table, on), and *gripped*
# when the closed fingers are around it. Holding needs both gripped and lifted off its support, so
# grasping skills (pickup, unstack) succeed only once the block is lifted, and gripper-empty needs the
# fingers off every block, so placing skills (putdown, stack) succeed only once the block is released.


def gripping(state: FetchState, block: str) -> bool:
    """The closed fingers are around the block, whether or not it is lifted (not a predicate)."""
    return (
        np.linalg.norm(state.gripper_position - state.block_positions[block]) <= GRASP_DISTANCE
        and state.finger_width < OPEN_FINGER_WIDTH
    )


def holding(state: FetchState, block: str) -> bool:
    """Gripped and lifted off the table and every other block."""
    return (
        gripping(state, block)
        and not on_table(state, block)
        and not any(on(state, block, other) for other in state.block_positions if other != block)
    )


def on_table(state: FetchState, block: str) -> bool:
    return abs(state.block_positions[block][2] - TABLE_REST_Z) <= Z_TOLERANCE


def on(state: FetchState, top: str, bottom: str) -> bool:
    """top rests on bottom: centred over it within a half block, one block height above it."""
    top_position, bottom_position = state.block_positions[top], state.block_positions[bottom]
    return (
        np.linalg.norm(top_position[:2] - bottom_position[:2]) <= BLOCK_HALF_SIZE
        and abs(top_position[2] - bottom_position[2] - 2 * BLOCK_HALF_SIZE) <= Z_TOLERANCE
    )


def clear(state: FetchState, block: str) -> bool:
    """Nothing on the block and not held (a held block is not clear in blocksworld)."""
    return not holding(state, block) and not any(
        on(state, other, block) for other in state.block_positions if other != block
    )


def raised(state: FetchState, block: str) -> bool:
    """Held with its centre within the RAISED_HEIGHT band above where it rests on the table."""
    height = state.block_positions[block][2] - TABLE_REST_Z
    return holding(state, block) and RAISED_HEIGHT[0] <= height <= RAISED_HEIGHT[1]


def beside(state: FetchState, block: str, other: str) -> bool:
    """Both blocks rest on the table, side by side: centres BESIDE_DISTANCE apart (blocks are 0.05 wide,
    and reset places blocks at least 0.10 apart, so no two blocks start beside each other)."""
    distance = np.linalg.norm(state.block_positions[block][:2] - state.block_positions[other][:2])
    return (
        on_table(state, block)
        and on_table(state, other)
        and BESIDE_DISTANCE[0] <= distance < BESIDE_DISTANCE[1]
    )


# Global table regions, so pushes have fixed goals (no relative predicates): a block resting on the table
# between REGION_DISTANCE[0] and REGION_DISTANCE[1] from the centre of the reset area (where blocks are
# placed, around the gripper's start), in one direction: a band, so a push has to stop in it, not just cross
# a line. Robot frame: front is toward the robot (smaller x), left is +y. push_goal() (and so the scripted
# push and the push reward) aims at the middle of the band.
REGION_CENTRE = (1.34, 0.75)
REGION_DISTANCE = (0.10, 0.18)
REGIONS = {  # name -> (axis, sign)
    "at-left": (1, +1), "at-right": (1, -1), "at-back": (0, +1), "at-front": (0, -1),
}


def in_region(state: FetchState, block: str, region: str) -> bool:
    axis, sign = REGIONS[region]
    position = state.block_positions[block]
    distance = sign * (position[axis] - REGION_CENTRE[axis])
    return on_table(state, block) and REGION_DISTANCE[0] <= distance <= REGION_DISTANCE[1]


def at_left(state: FetchState, block: str) -> bool:
    return in_region(state, block, "at-left")


def at_right(state: FetchState, block: str) -> bool:
    return in_region(state, block, "at-right")


def at_front(state: FetchState, block: str) -> bool:
    return in_region(state, block, "at-front")


def at_back(state: FetchState, block: str) -> bool:
    return in_region(state, block, "at-back")


def push_goal(state: FetchState, grounded_action_model) -> np.ndarray:
    """Where a push should bring the moved block (first argument), from its effects: the middle of each
    region's band (other coordinates unchanged), or BESIDE_SPACING from ?u (second argument) for (beside ?o ?u).
    Shared by the scripted push policy and the push reward."""
    block = grounded_action_model.arguments[0]
    goal = state.block_positions[block][:2].astype(np.float64).copy()
    for effect in grounded_action_model.effects:
        if effect.delete:
            continue
        name = effect.predicate.name
        if name in REGIONS:
            axis, sign = REGIONS[name]
            goal[axis] = REGION_CENTRE[axis] + sign * float(np.mean(REGION_DISTANCE))
        elif name == "beside":
            other = state.block_positions[grounded_action_model.arguments[1]][:2]
            away = goal - other
            goal = other + BESIDE_SPACING * away / max(np.linalg.norm(away), 1e-6)
    return goal


def gripper_empty(state: FetchState) -> bool:
    """The fingers are around no block (open, or closed on nothing)."""
    return not any(gripping(state, block) for block in state.block_positions)


FETCH_PREDICATES: dict[str, Callable[..., bool]] = {
    "holding": holding,
    "on-table": on_table,
    "on": on,
    "clear": clear,
    "gripper-empty": gripper_empty,
    "raised": raised,
    "beside": beside,
    "at-left": at_left,
    "at-right": at_right,
    "at-front": at_front,
    "at-back": at_back,
}

# The frame: predicates whose facts an action must leave unchanged unless its effects mention them. The
# geometric predicates (raised, beside, at-*) are left out: they change as side effects of the ones listed
# (a block picked up leaves its region), which the operators do not spell out.
FETCH_FRAME_PREDICATES = frozenset({"on-table", "on", "clear", "holding", "gripper-empty"})

# Failure: a state in which an episode ends unsuccessfully, whatever its effects; the same for every skill.
DROP_HEIGHT = 0.02  # a block let go more than this above its support was dropped, not placed
DISTURB_DISTANCE = 0.03  # another block moved further than this from its start was knocked or pushed aside
# Net contact force on the gripper (FetchState.gripper_force). Holding a block is ~20 N, lifting or placing one
# peaks ~55 N, pushing one ~65 N. Above SAFE_GRIPPER_FORCE the shaped reward penalizes the excess, up to
# MAX_GRIPPER_FORCE (FetchMotionReward). Above MAX_GRIPPER_FORCE for FORCE_FAILURE_STEPS consecutive steps
# (0.2 s) the episode fails: pressing on a block or the table, which stays above it (pressing into the table:
# ~55 kN on every step). Shorter stretches are penalized, not failures: first contact of the position-controlled
# gripper spikes for one step (ramming a block with closed fingers: ~900 N once; random exploration: up to
# ~38 kN), and random exploration bumps the table for a few steps (over 500 N for 2+ steps in about half of
# random episodes, for 5+ in about 1 in 8, never for 10).
SAFE_GRIPPER_FORCE = 100.0
MAX_GRIPPER_FORCE = 500.0
FORCE_FAILURE_STEPS = 5


def dropped(grounded_action_model, start_info: dict, info: dict) -> bool:
    """The moved block (first argument) is not gripped and its centre is more than DROP_HEIGHT above where it
    would rest on its support: the table, or the top of a block under it. It was let go in the air (it is
    falling), or pushed off the table."""
    state = info["environment_state"]
    block = grounded_action_model.arguments[0]
    position = state.block_positions[block]
    rest_heights = [TABLE_REST_Z] + [  # where the block's centre would rest on the table or on a block below
        other[2] + 2 * BLOCK_HALF_SIZE for name, other in state.block_positions.items()
        if name != block and other[2] < position[2] and np.linalg.norm(other[:2] - position[:2]) < 2 * BLOCK_HALF_SIZE
    ]
    return not gripping(state, block) and position[2] - max(rest_heights) > DROP_HEIGHT


def disturbed(grounded_action_model, start_info: dict, info: dict) -> str | None:
    """A block other than the moved one (first argument) that is more than DISTURB_DISTANCE from where it was
    when the episode started (knocked over or pushed aside), or None. Every skill here moves one block."""
    moved = grounded_action_model.arguments[0]
    start = start_info["environment_state"].block_positions
    now = info["environment_state"].block_positions
    return next((name for name in now if name != moved and np.linalg.norm(now[name] - start[name]) > DISTURB_DISTANCE),
                None)


# Settled success needs the moved block (first argument) to move less than this per step on every step the effects
# hold, so a block passing slowly through a goal region (e.g. lifted on and on through the raised band) has not
# stopped there. Held blocks jitter well under it.
AT_REST_DISTANCE = 0.002


def moved_block_at_rest(grounded_action_model, previous_info: dict, info: dict) -> bool:
    """The moved block (first argument) moved less than AT_REST_DISTANCE since the previous step
    (SkillEnvironment's at_rest)."""
    block = grounded_action_model.arguments[0]
    before = previous_info["environment_state"].block_positions[block]
    now = info["environment_state"].block_positions[block]
    return float(np.linalg.norm(np.asarray(now) - np.asarray(before))) < AT_REST_DISTANCE


def check_fetch_failure(grounded_action_model, start_info: dict, info: dict) -> str | None:
    """Checks whether the episode has failed in this step: returns the reason if it has, None if it has not.
    Called on every step by SkillEnvironment (its check_failure); a returned reason makes the step a failure,
    which ends the episode. The failures, checked in this order: an extreme force on the gripper for
    FORCE_FAILURE_STEPS consecutive steps, the moved block dropped, another block disturbed."""
    recent = info["environment_state"].gripper_force_history[-FORCE_FAILURE_STEPS:]
    if len(recent) == FORCE_FAILURE_STEPS and min(recent) > MAX_GRIPPER_FORCE:
        return f"excessive force on the gripper (over {min(recent):.0f} N for {FORCE_FAILURE_STEPS} steps)"
    if dropped(grounded_action_model, start_info, info):
        return f"dropped {grounded_action_model.arguments[0]}"
    other = disturbed(grounded_action_model, start_info, info)
    return f"disturbed {other}" if other else None


FETCH_PREDICATE_ARITIES: dict[str, int] = {
    name: len(inspect.signature(function).parameters) - 1 for name, function in FETCH_PREDICATES.items()
}


class FetchPredicateEvaluationWrapper(gym.Wrapper):
    """Adds info["facts"]: the ground predicates true in info["environment_state"].

    Wraps an environment that already provides info["environment_state"], i.e.
        FetchPredicateEvaluationWrapper(FetchEnvStateAnnotationWrapper(gym.make(...)))
    Predicate names and arities match those used in the symbolic action models.
    Observations are passed through unchanged.
    """

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.facts = self.evaluate(info["environment_state"])
        logger.info("facts at reset: %s", format_facts(self.facts))
        return obs, {**info, "facts": self.facts}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        facts = self.evaluate(info["environment_state"])
        if facts != self.facts:
            logger.info(
                "facts added: %s  removed: %s", format_facts(facts - self.facts), format_facts(self.facts - facts)
            )
            self.facts = facts
        return obs, reward, terminated, truncated, {**info, "facts": facts}

    @staticmethod
    def evaluate(state: FetchState) -> frozenset[Predicate]:
        """All ground predicates true in state (closed world), over distinct blocks."""
        blocks = list(state.block_positions)
        return frozenset( # This eval is computationally heavy, but necessary
            Predicate(name, arguments)
            for name, function in FETCH_PREDICATES.items()
            for arguments in permutations(blocks, FETCH_PREDICATE_ARITIES[name])
            if function(state, *arguments)
        )
