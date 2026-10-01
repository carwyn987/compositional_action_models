"""Scripted (hardcoded) Fetch policies for pickup and putdown.

Both read the target block from the grounded action's first argument and the
scene from info["environment_state"] (FetchState). Actions are
[dx, dy, dz, gripper] in [-1, 1]; gripper > 0 opens, < 0 closes.
"""

import numpy as np

from cam.environments.fetch.fetch_predicate_evaluation_wrapper import TABLE_REST_Z
from cam.policies.policy import Policy

OPEN, CLOSE = 1.0, -1.0
GAIN = 10.0  # proportional gain from position error to action


def servo(gripper: np.ndarray, target: np.ndarray, gripper_command: float) -> np.ndarray:
    """Proportional step toward target (clipped to [-1, 1]), with the given gripper command."""
    delta = np.clip(GAIN * (target - gripper), -1.0, 1.0)
    return np.array([*delta, gripper_command], dtype=np.float32)


class FetchScriptedPickupPolicy(Policy):
    """Approach above the target block, descend, close, lift."""

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
