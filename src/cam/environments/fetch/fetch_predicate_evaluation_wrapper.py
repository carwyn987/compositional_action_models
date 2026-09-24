"""Wrapper that evaluates symbolic predicates on the annotated Fetch state.

FETCH_PREDICATES maps each predicate name (as written in the symbolic action
models) to a function `(state, *objects) -> bool`. The number of object
parameters after `state` is the predicate's arity. evaluate() grounds every
predicate over every ordered tuple of distinct blocks in the scene, so the same
definitions serve any number of blocks.
"""

import inspect
from itertools import permutations
from typing import Callable

import gymnasium as gym
import numpy as np

from cam.domain.operators.pddl import Predicate
from cam.environments.fetch.fetch_env_state_annotation_wrapper import FetchState
from cam.environments.fetch.fetch_multiblock_environment import BLOCK_HALF_SIZE

TABLE_REST_Z = 0.425  # block centre height when resting on the table
Z_TOLERANCE = 0.01  # height tolerance for resting on the table / on another block
LIFT_THRESHOLD = 0.01  # height above rest at which a block near the gripper counts as held
GRASP_DISTANCE = 0.03  # max gripper-to-block-centre distance for holding
OPEN_FINGER_WIDTH = 0.07  # finger width at or above which the gripper holds nothing


def holding(state: FetchState, block: str) -> bool:
    position = state.block_positions[block]
    return (
        np.linalg.norm(state.gripper_position - position) <= GRASP_DISTANCE
        and position[2] - TABLE_REST_Z >= LIFT_THRESHOLD
        and state.finger_width < OPEN_FINGER_WIDTH
    )


def on_table(state: FetchState, block: str) -> bool:
    height = state.block_positions[block][2]
    return abs(height - TABLE_REST_Z) <= Z_TOLERANCE and not holding(state, block)


def on(state: FetchState, top: str, bottom: str) -> bool:
    top_position, bottom_position = state.block_positions[top], state.block_positions[bottom]
    return (
        np.linalg.norm(top_position[:2] - bottom_position[:2]) <= BLOCK_HALF_SIZE
        and abs(top_position[2] - bottom_position[2] - 2 * BLOCK_HALF_SIZE) <= Z_TOLERANCE
        and not holding(state, top)
    )


def clear(state: FetchState, block: str) -> bool:
    """Nothing on the block and not held (a held block is not clear in blocksworld)."""
    return not holding(state, block) and not any(
        on(state, other, block) for other in state.block_positions if other != block
    )


def gripper_empty(state: FetchState) -> bool:
    return not any(holding(state, block) for block in state.block_positions)


FETCH_PREDICATES: dict[str, Callable[..., bool]] = {
    "holding": holding,
    "on-table": on_table,
    "on": on,
    "clear": clear,
    "gripper-empty": gripper_empty,
}

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
        return obs, {**info, "facts": self.evaluate(info["environment_state"])}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = {**info, "facts": self.evaluate(info["environment_state"])}
        return obs, reward, terminated, truncated, info

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
