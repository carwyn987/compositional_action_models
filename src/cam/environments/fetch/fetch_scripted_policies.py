"""Scripted (hardcoded) Fetch policies for pickup, putdown, stack and place-beside (unstack and the
raised variants use the pickup policy, whose motion is relative to the block and lifts high enough).

Each reads the moved block from the grounded action's first argument (stack and
unstack: the other block from the second) and the scene from
info["environment_state"] (FetchState). Actions are
[dx, dy, dz, gripper] in [-1, 1]; gripper > 0 opens, < 0 closes.
"""

import numpy as np

from cam.environments.fetch.fetch_multiblock_environment import BLOCK_HALF_SIZE
from cam.environments.fetch.fetch_predicate_evaluation_wrapper import RAISED_HEIGHT, TABLE_REST_Z, push_goal
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
        return servo(gripper, self.lift_target(), CLOSE)

    def lift_target(self) -> np.ndarray:
        """Where to lift the gripper (and block) to: LIFT_HEIGHT above where the block started."""
        return self.block_start + [0.0, 0.0, self.LIFT_HEIGHT]



class FetchScriptedPickupRaisedPolicy(FetchScriptedPickupPolicy):
    """The pickup motion, lifting the block to the middle of the raised band wherever it started (for
    pickup-raised and unstack-raised)."""

    def lift_target(self) -> np.ndarray:
        return np.array([self.block_start[0], self.block_start[1], TABLE_REST_Z + float(np.mean(RAISED_HEIGHT))])

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
    stack motion, landing at SPACING from the other block along x, on the side the held block comes from.
    Along x because the fingers open along y: landing along y, opening them would knock the other block away."""

    SPACING = 0.075  # centre-to-centre, inside beside's range

    def reset(self) -> None:
        super().reset()
        self.spot = None

    def resting_position(self, state, block: np.ndarray, base: np.ndarray) -> np.ndarray:
        if self.spot is None:  # fixed for the episode, from where the block starts
            side = 1.0 if block[0] >= base[0] else -1.0
            self.spot = np.array([base[0] + side * self.SPACING, base[1], TABLE_REST_Z])
        return self.spot


class FetchScriptedPushPolicy(Policy):
    """Push the block (first argument) along the table to its goal (push_goal: a region, or beside the second
    argument): move above a point behind the block (opposite the goal), lower to the block's height with the
    gripper closed, then push toward the goal and stop there. A goal that differs along both axes (the
    diagonal) is reached in two straight pushes, x then y: pushing a cube diagonally hits a corner and spins it.
    While pushing it steers toward the goal, so a block that drifts sideways is pushed back on course."""

    BEHIND = BLOCK_HALF_SIZE + 0.03  # gripper centre this far behind the block centre while pushing
    HOVER = 0.08
    PUSH_SPEED = 0.25  # action limit while pushing: at full speed the gripper rams the block (~650 N contact force)
    REACHED_TOLERANCE = 0.01
    LEG_DONE = 0.025  # block within this of a leg's goal along the leg (goals are mid-band, 4 cm inside their region)

    def reset(self) -> None:
        self.legs: list[np.ndarray] | None = None

    def __call__(self, obs, info, grounded_action_model) -> np.ndarray:
        state = info["environment_state"]
        gripper = state.gripper_position
        block = state.block_positions[grounded_action_model.arguments[0]]
        if self.legs is None:  # fixed for the episode
            goal = push_goal(state, grounded_action_model)
            corner = np.array([goal[0], block[1]])
            two_axes = abs(goal[0] - block[0]) > 0.02 and abs(goal[1] - block[1]) > 0.02
            self.legs = [corner, goal] if two_axes else [goal]
            self.start_leg()
        goal = self.legs[0]
        if self.direction is None:
            offset = goal - block[:2]
            self.direction = offset / max(np.linalg.norm(offset), 1e-6)
        if len(self.legs) > 1 and np.dot(goal - block[:2], self.direction) < self.LEG_DONE:  # leg done
            self.legs.pop(0)
            self.start_leg()
            goal = self.legs[0]
            offset = goal - block[:2]
            self.direction = offset / max(np.linalg.norm(offset), 1e-6)
        behind = np.array([*(block[:2] - self.BEHIND * self.direction), block[2]])
        if self.phase == "rise":  # straight up first, so moving to the next leg's start does not sweep the block
            target = np.array([gripper[0], gripper[1], block[2] + self.HOVER])
            if gripper[2] >= block[2] + self.HOVER - self.REACHED_TOLERANCE:
                self.phase = "approach"
        if self.phase == "approach":
            target = behind + [0.0, 0.0, self.HOVER]
            if np.linalg.norm(gripper - target) < self.REACHED_TOLERANCE:
                self.phase = "descend"
        if self.phase == "descend":
            target = behind
            if np.linalg.norm(gripper - target) < self.REACHED_TOLERANCE:
                self.phase = "push"
        if self.phase == "push":
            # Aim just ahead of the point behind the block, along the block -> goal direction recomputed each step:
            # pushes through the block's centre and steers it back if it drifts sideways; slows down at the goal.
            to_goal = goal - block[:2]
            distance = float(np.linalg.norm(to_goal))
            heading = to_goal / distance if distance > 0.005 else self.direction
            if distance > 0.02 and np.dot(block[:2] - gripper[:2], heading) < 0.02:
                # The gripper is no longer behind the block (it overshot the goal or slid past): lift and
                # come back from behind, instead of pushing it the wrong way.
                self.phase, self.direction = "rise", heading
                return servo(gripper, np.array([gripper[0], gripper[1], block[2] + self.HOVER]), CLOSE)
            target = np.array([*(block[:2] - self.BEHIND * heading + min(distance, 0.05) * heading), block[2]])
            action = servo(gripper, target, CLOSE)
            action[:3] = np.clip(action[:3], -self.PUSH_SPEED, self.PUSH_SPEED)
            return action
        return servo(gripper, target, CLOSE)

    def start_leg(self) -> None:
        self.phase, self.direction = "rise", None
