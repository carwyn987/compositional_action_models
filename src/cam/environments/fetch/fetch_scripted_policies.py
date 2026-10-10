"""Scripted (hardcoded) Fetch policies for pickup, putdown, stack and place-beside (unstack and the
raised variants use the pickup policy, whose motion is relative to the block and lifts high enough).

Each reads the moved block from the grounded action's first argument (stack and
unstack: the other block from the second) and the scene from
info["environment_state"] (FetchState). Actions are
[dx, dy, dz, gripper] in [-1, 1]; gripper > 0 opens, < 0 closes.
"""

import numpy as np

from cam.environments.fetch.fetch_multiblock_environment import BLOCK_HALF_SIZE
from cam.environments.fetch.fetch_predicate_evaluation_wrapper import TABLE_REST_Z
from cam.policies.policy import Policy

OPEN, CLOSE = 1.0, -1.0
GAIN = 10.0  # proportional gain from position error to action


def servo(gripper: np.ndarray, target: np.ndarray, gripper_command: float) -> np.ndarray:
    """Proportional step toward target (clipped to [-1, 1]), with the given gripper command."""
    delta = np.clip(GAIN * (target - gripper), -1.0, 1.0)
    return np.array([*delta, gripper_command], dtype=np.float32)


class FetchScriptedPickupPolicy(Policy):
    """Approach above the target block, descend, close, lift. Every target is relative to the block's
    position, so the same motion unstacks a block resting on another."""

    APPROACH_HEIGHT = 0.10
    LIFT_HEIGHT = 0.15
    REACHED_TOLERANCE = 0.01
    CLOSE_STEPS = 10

    def reset(self) -> None:
        self.phase = "approach"
        self.close_steps_taken = 0
        self.block_start = None

    def __call__(self, obs, info, grounded_action_model) -> np.ndarray:
        state = info["environment_state"]
        gripper = state.gripper_position
        block = state.block_positions[grounded_action_model.arguments[0]]
        if self.block_start is None:
            self.block_start = block.copy()

        if self.phase == "approach":
            target = block + [0.0, 0.0, self.APPROACH_HEIGHT]
            if np.linalg.norm(gripper - target) < self.REACHED_TOLERANCE:
                self.phase = "descend"
            return servo(gripper, target, OPEN)
        if self.phase == "descend":
            if np.linalg.norm(gripper - block) < self.REACHED_TOLERANCE:
                self.phase = "close"
            return servo(gripper, block, OPEN)
        if self.phase == "close":
            self.close_steps_taken += 1
            if self.close_steps_taken >= self.CLOSE_STEPS:
                self.phase = "lift"
            return np.array([0.0, 0.0, 0.0, CLOSE], dtype=np.float32)
        return servo(gripper, self.block_start + [0.0, 0.0, self.LIFT_HEIGHT], CLOSE)


class FetchScriptedPutdownPolicy(Policy):
    """Lower the held target block to the table, then open the gripper."""

    RELEASE_HEIGHT = 0.003  # block height above table rest at which to open

    def reset(self) -> None:
        self.releasing = False

    def __call__(self, obs, info, grounded_action_model) -> np.ndarray:
        state = info["environment_state"]
        gripper = state.gripper_position
        height_above_table = state.block_positions[grounded_action_model.arguments[0]][2] - TABLE_REST_Z
        if height_above_table <= self.RELEASE_HEIGHT:
            self.releasing = True
        if self.releasing:
            return np.array([0.0, 0.0, 0.0, OPEN], dtype=np.float32)
        return servo(gripper, gripper - [0.0, 0.0, height_above_table], CLOSE)


class FetchScriptedStackPolicy(Policy):
    """Carry the held block (first argument) up to clear the base block (second argument), move it over
    the base, lower it until it rests on the base, then open the gripper.

    The gripper is steered so that the *block* reaches each target: the gripper target is the gripper
    position plus the block's offset from its target, so a block held off-centre still lands centred.
    """

    CARRY_CLEARANCE = 0.05  # gap between the held block's bottom and the base's top while moving over it
    ALIGNED_TOLERANCE = 0.003  # horizontal block-to-base distance at which to start lowering
    RELEASE_GAP = 0.003  # block height above its resting height on the base at which to open

    def reset(self) -> None:
        self.phase = "carry"

    def resting_position(self, state, block: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Where the held block's centre should end up: on top of the base."""
        return base + [0.0, 0.0, 2 * BLOCK_HALF_SIZE]

    def __call__(self, obs, info, grounded_action_model) -> np.ndarray:
        state = info["environment_state"]
        gripper = state.gripper_position
        block, base = (state.block_positions[name] for name in grounded_action_model.arguments[:2])
        resting = self.resting_position(state, block, base)
        carry_height = base[2] + 2 * BLOCK_HALF_SIZE + self.CARRY_CLEARANCE  # held block's centre clears the base

        if self.phase == "carry":  # straight up first, so the move over the base does not hit it
            target = np.array([block[0], block[1], carry_height])
            if block[2] >= carry_height - 0.005:
                self.phase = "align"
        if self.phase == "align":
            target = np.array([resting[0], resting[1], carry_height])
            if np.linalg.norm(block[:2] - resting[:2]) < self.ALIGNED_TOLERANCE:
                self.phase = "lower"
        if self.phase == "lower":
            target = resting
            if block[2] - resting[2] <= self.RELEASE_GAP:
                self.phase = "release"
        if self.phase == "release":
            return np.array([0.0, 0.0, 0.0, OPEN], dtype=np.float32)
        return servo(gripper, gripper + (target - block), CLOSE)



class FetchScriptedPlaceBesidePolicy(FetchScriptedStackPolicy):
    """Place the held block (first argument) on the table next to the other block (second argument): the
    stack motion, landing at SPACING from the other block on the side the held block comes from."""

    SPACING = 0.075  # centre-to-centre, inside beside's range

    def reset(self) -> None:
        super().reset()
        self.spot = None

    def resting_position(self, state, block: np.ndarray, base: np.ndarray) -> np.ndarray:
        if self.spot is None:  # fixed for the episode, from where the block starts
            direction = block[:2] - base[:2]
            direction = direction / max(np.linalg.norm(direction), 1e-6)
            self.spot = np.array([*(base[:2] + self.SPACING * direction), TABLE_REST_Z])
        return self.spot
