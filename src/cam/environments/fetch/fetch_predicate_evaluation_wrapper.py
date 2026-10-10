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


def gripper_empty(state: FetchState) -> bool:
    """The fingers are around no block (open, or closed on nothing)."""
    return not any(gripping(state, block) for block in state.block_positions)


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
