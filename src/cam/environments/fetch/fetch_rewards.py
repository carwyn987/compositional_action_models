"""Shaped Fetch rewards (ported from deprecated/fetch_blockworld/rewards/).

Each reads the moved block from the grounded action's first argument (stack: the
base block from the second) and the scene from info["environment_state"]
(FetchState) and info["facts"]. Every reward is staged: each stage has dense
progress terms (the change since the previous step, clipped) and a one-time
bonus, and every term is reported as a named reward component.
"""

from dataclasses import dataclass, field

import numpy as np

from cam.domain.symbols import Predicate
from cam.environments.fetch.fetch_multiblock_environment import BLOCK_HALF_SIZE
from cam.environments.fetch.fetch_predicate_evaluation_wrapper import (
    BESIDE_DISTANCE,
    OPEN_FINGER_WIDTH,
    RAISED_HEIGHT,
    TABLE_REST_Z,
)
from cam.rewards.reward_function import RewardFunction


@dataclass(frozen=True)
class FetchPickupRewardConfig:
    # General costs
    time_penalty: float = 0.01
    action_l2_penalty: float = 0.001

    # 1. Approach object
    reach_progress_weight: float = 15.0
    reach_dense_weight: float = 0.10

    # 2. Get above object
    above_xy_tolerance: float = 0.030
    above_min_clearance: float = 0.040
    above_bonus: float = 1.5

    # 3. Lower into grasp pose
    grasp_xy_tolerance: float = 0.020
    grasp_z_offset: float = -0.005
    grasp_z_tolerance: float = 0.012
    descend_progress_weight: float = 20.0
    grasp_pose_bonus: float = 2.0

    # 4. Close/grip
    close_progress_weight: float = 10.0
    grip_bonus: float = 4.0

    # 5. Lift object (off the table; for unstack, off the block below)
    target_lift: float = 0.06
    lift_off_height: float = 0.010
    lift_off_bonus: float = 2.0
    lift_progress_weight: float = 8.0

    # 6. Complete pickup
    success_bonus: float = 15.0

    # Prevent single noisy transitions from producing huge rewards.
    max_position_progress: float = 0.025
    max_lift_fraction_progress: float = 0.25


@dataclass
class FetchPickupReward(RewardFunction):
    """Staged pickup shaping: approach, get above, descend, close, lift, success.

    Progress terms reward the change since the previous step; each stage's bonus
    is paid once per episode. The lift is measured from rest_height (the table).
    """

    config: FetchPickupRewardConfig = field(default_factory=FetchPickupRewardConfig)

    def __post_init__(self):
        self.reset(None, {})

    def rest_height(self, grounded_action_model, info) -> float:
        """Height of the block's centre where it rests before the lift."""
        return TABLE_REST_Z

    def reset(self, grounded_action_model, info) -> None:
        self.lift_reference = self.rest_height(grounded_action_model, info) if grounded_action_model else TABLE_REST_Z
        self.previous_distance: float | None = None
        self.previous_grasp_error: float | None = None
        self.previous_gripper_width: float | None = None
        self.previous_lift_fraction = 0.0
        self.above_earned = False
        self.grasp_pose_earned = False
        self.grip_earned = False
        self.lift_off_earned = False
        self.success_earned = False

    def __call__(self, grounded_action_model, action, info, success):
        cfg = self.config
        state = info["environment_state"]
        block = grounded_action_model.arguments[0]
        grip_pos = np.asarray(state.gripper_position, dtype=np.float64)
        object_pos = np.asarray(state.block_positions[block], dtype=np.float64)
        gripper_width = float(state.finger_width)
        action = np.asarray(action, dtype=np.float64)

        distance = float(np.linalg.norm(grip_pos - object_pos))
        xy_distance = float(np.linalg.norm(grip_pos[:2] - object_pos[:2]))
        z_clearance = float(grip_pos[2] - object_pos[2])
        grasp_z_error = abs(float(grip_pos[2] - (object_pos[2] + cfg.grasp_z_offset)))
        # Includes XY alignment so moving sideways away from the block is penalized.
        grasp_error = float(np.sqrt(xy_distance**2 + grasp_z_error**2))
        lift = max(0.0, float(object_pos[2] - self.lift_reference))
        lift_fraction = float(np.clip(lift / max(cfg.target_lift, 1e-6), 0.0, 1.0))

        above_now = xy_distance <= cfg.above_xy_tolerance and z_clearance >= cfg.above_min_clearance
        at_grasp_pose = xy_distance <= cfg.grasp_xy_tolerance and grasp_z_error <= cfg.grasp_z_tolerance
        # Forgiving: holding, or the gripper closing while correctly positioned around the block.
        gripping_now = Predicate("holding", (block,)) in info["facts"] or (
            at_grasp_pose and gripper_width < OPEN_FINGER_WIDTH
        )
        lift_off_now = lift >= cfg.lift_off_height

        components = {
            "time": -cfg.time_penalty,
            "movement": -cfg.action_l2_penalty * float(np.dot(action, action)),
            "reach": 0.0,
            "above": 0.0,
            "descend": 0.0,
            "close": 0.0,
            "grip": 0.0,
            "lift_off": 0.0,
            "lift": 0.0,
            "success": 0.0,
        }

        # 1. Approach the block: reward reduced distance, plus a small proximity signal.
        if not self.above_earned:
            if self.previous_distance is not None:
                progress = np.clip(
                    self.previous_distance - distance, -cfg.max_position_progress, cfg.max_position_progress
                )
                components["reach"] += cfg.reach_progress_weight * float(progress)
            components["reach"] += cfg.reach_dense_weight * np.exp(-15.0 * distance)

        # 2. One-time reward for becoming aligned over the block.
        if above_now and not self.above_earned:
            components["above"] = cfg.above_bonus
            self.above_earned = True

        # 3. Reward lowering toward the grasp pose.
        if self.above_earned and not self.grasp_pose_earned:
            if self.previous_grasp_error is not None:
                progress = np.clip(
                    self.previous_grasp_error - grasp_error, -cfg.max_position_progress, cfg.max_position_progress
                )
                components["descend"] = cfg.descend_progress_weight * float(progress)
            if at_grasp_pose:
                components["descend"] += cfg.grasp_pose_bonus
                self.grasp_pose_earned = True

        # 4. Reward closing only after reaching the grasp pose.
        if self.grasp_pose_earned and not self.grip_earned:
            if self.previous_gripper_width is not None:
                close_progress = np.clip(self.previous_gripper_width - gripper_width, -0.02, 0.02)
                components["close"] = cfg.close_progress_weight * float(close_progress)
            if gripping_now:
                components["grip"] = cfg.grip_bonus
                self.grip_earned = True

        # 5. Reward upward block movement after gripping.
        if self.grip_earned or lift > 0.0:
            lift_progress = np.clip(
                lift_fraction - self.previous_lift_fraction,
                -cfg.max_lift_fraction_progress,
                cfg.max_lift_fraction_progress,
            )
            components["lift"] = cfg.lift_progress_weight * float(lift_progress)
        if lift_off_now and not self.lift_off_earned:
            components["lift_off"] = cfg.lift_off_bonus
            self.lift_off_earned = True

        # 6. One-time terminal success reward.
        if success and not self.success_earned:
            components["success"] = cfg.success_bonus
            self.success_earned = True

        self.previous_distance = distance
        self.previous_grasp_error = grasp_error
        self.previous_gripper_width = gripper_width
        self.previous_lift_fraction = lift_fraction
        components = {name: float(value) for name, value in components.items()}
        return sum(components.values()), components


@dataclass
class FetchUnstackReward(FetchPickupReward):
    """Unstack is pickup from on top of another block: the same stages (approach, above, descend, close,
    lift, success), with the lift measured from where the block rests on the block below."""

    def rest_height(self, grounded_action_model, info) -> float:
        if "environment_state" not in info:
            return TABLE_REST_Z + 2 * BLOCK_HALF_SIZE  # resting on a block on the table
        return float(info["environment_state"].block_positions[grounded_action_model.arguments[0]][2])


@dataclass(frozen=True)
class FetchStackRewardConfig:
    # General costs
    time_penalty: float = 0.01
    action_l2_penalty: float = 0.001

    # 1. Carry: raise the held block until its bottom clears the base's top by carry_clearance
    carry_clearance: float = 0.05
    carry_tolerance: float = 0.005
    carry_progress_weight: float = 10.0
    carried_bonus: float = 2.0

    # 2. Align the held block horizontally over the base
    align_progress_weight: float = 12.0
    aligned_xy_tolerance: float = 0.010
    aligned_bonus: float = 2.0

    # 3. Lower gently until the block rests on the base
    descend_progress_weight: float = 18.0
    gentle_descent_speed: float = 0.012  # per step; faster descent is penalized
    gentle_descent_penalty: float = 25.0
    placed_bonus: float = 4.0

    # 4. Release the block
    release_progress_weight: float = 8.0
    release_bonus: float = 3.0

    # 5. Complete stack
    success_bonus: float = 15.0

    max_position_progress: float = 0.025


@dataclass
class FetchStackReward(RewardFunction):
    """Staged stack shaping, starting with the block held (setup picks it up): carry it above the base,
    align over the base, lower gently onto it, release, success. Ported from
    deprecated/fetch_blockworld/rewards/stack_rewards.py without its approach and grasp stages, which
    belong to pickup here.

    Carry and align progress count only while the block is held, so dropping it earns nothing. Progress
    terms reward the change since the previous step; each stage's bonus is paid once per episode.
    """

    config: FetchStackRewardConfig = field(default_factory=FetchStackRewardConfig)

    def __post_init__(self):
        self.reset(None, {})

    def reset(self, grounded_action_model, info) -> None:
        self.previous_carry_gap: float | None = None
        self.previous_align_distance: float | None = None
        self.previous_place_gap: float | None = None
        self.previous_block_z: float | None = None
        self.previous_gripper_width: float | None = None
        self.carried_earned = False
        self.aligned_earned = False
        self.placed_earned = False
        self.release_earned = False
        self.success_earned = False

    def resting_z(self, block: np.ndarray, base: np.ndarray) -> float:
        """Height of the block's centre once placed: on top of the base."""
        return float(base[2] + 2 * BLOCK_HALF_SIZE)

    def align_distance(self, block: np.ndarray, base: np.ndarray) -> float:
        """Horizontal distance from the block to where it should be placed: over the base."""
        return float(np.linalg.norm(block[:2] - base[:2]))

    def placed(self, facts: frozenset, block_name: str, base_name: str) -> bool:
        return Predicate("on", (block_name, base_name)) in facts

    def progress(self, previous: float | None, current: float, weight: float) -> float:
        """weight x the reduction of a gap since the previous step, clipped."""
        if previous is None:
            return 0.0
        limit = self.config.max_position_progress
        return weight * float(np.clip(previous - current, -limit, limit))

    def __call__(self, grounded_action_model, action, info, success):
        cfg = self.config
        state = info["environment_state"]
        block_name, base_name = grounded_action_model.arguments[:2]
        block = np.asarray(state.block_positions[block_name], dtype=np.float64)
        base = np.asarray(state.block_positions[base_name], dtype=np.float64)
        gripper_width = float(state.finger_width)
        action = np.asarray(action, dtype=np.float64)

        resting_z = self.resting_z(block, base)
        carry_z = float(base[2] + 2 * BLOCK_HALF_SIZE + cfg.carry_clearance)  # the held block clears the base
        carry_gap = max(0.0, carry_z - float(block[2]))
        align_distance = self.align_distance(block, base)
        place_gap = abs(float(block[2]) - resting_z)
        held = Predicate("holding", (block_name,)) in info["facts"]
        placed = self.placed(info["facts"], block_name, base_name)
        released = Predicate("gripper-empty", ()) in info["facts"]

        components = {
            "time": -cfg.time_penalty,
            "movement": -cfg.action_l2_penalty * float(np.dot(action, action)),
            "carry": 0.0,
            "align": 0.0,
            "descend": 0.0,
            "gentle": 0.0,
            "placed": 0.0,
            "release": 0.0,
            "success": 0.0,
        }

        # 1. Raise the held block high enough to clear the base.
        if held and not self.carried_earned:
            components["carry"] = self.progress(self.previous_carry_gap, carry_gap, cfg.carry_progress_weight)
            if carry_gap <= cfg.carry_tolerance:
                components["carry"] += cfg.carried_bonus
                self.carried_earned = True

        # 2. Move the held block over the base.
        if held and self.carried_earned and not self.aligned_earned:
            components["align"] = self.progress(
                self.previous_align_distance, align_distance, cfg.align_progress_weight
            )
            if align_distance <= cfg.aligned_xy_tolerance:
                components["align"] += cfg.aligned_bonus
                self.aligned_earned = True

        # 3. Lower onto the base, penalizing a fast drop.
        if self.aligned_earned and not self.placed_earned:
            components["descend"] = self.progress(self.previous_place_gap, place_gap, cfg.descend_progress_weight)
            if self.previous_block_z is not None:
                descent_speed = max(0.0, self.previous_block_z - float(block[2]))
                components["gentle"] = -cfg.gentle_descent_penalty * max(0.0, descent_speed - cfg.gentle_descent_speed)
            if placed:
                components["placed"] = cfg.placed_bonus
                self.placed_earned = True

        # 4. Open the gripper while the block rests on the base.
        if placed:
            if self.previous_gripper_width is not None:
                components["release"] = cfg.release_progress_weight * float(
                    np.clip(gripper_width - self.previous_gripper_width, -0.02, 0.02)
                )
            if released and not self.release_earned:
                components["release"] += cfg.release_bonus
                self.release_earned = True

        # 5. One-time terminal success reward.
        if success and not self.success_earned:
            components["success"] = cfg.success_bonus
            self.success_earned = True

        self.previous_carry_gap = carry_gap
        self.previous_align_distance = align_distance
        self.previous_place_gap = place_gap
        self.previous_block_z = float(block[2])
        self.previous_gripper_width = gripper_width
        components = {name: float(value) for name, value in components.items()}
        return sum(components.values()), components


@dataclass
class FetchPickupRaisedReward(FetchPickupReward):
    """pickup's stages with the lift target RAISED_HEIGHT + 1 cm above the table, so the lift stage leads
    to (raised ?o)."""

    config: FetchPickupRewardConfig = field(
        default_factory=lambda: FetchPickupRewardConfig(target_lift=RAISED_HEIGHT + 0.01)
    )


@dataclass
class FetchPlaceBesideReward(FetchStackReward):
    """stack's stages (carry, align, gentle descend, placed, release, success) with the block placed on the
    table at SPACING from the other block instead of on top of it; placed once (on-table ?o) and
    (beside ?o ?u) hold."""

    SPACING = float(np.mean(BESIDE_DISTANCE))  # centre-to-centre target, the middle of beside's range

    def resting_z(self, block: np.ndarray, base: np.ndarray) -> float:
        return TABLE_REST_Z

    def align_distance(self, block: np.ndarray, base: np.ndarray) -> float:
        """How far the block is from the ring at SPACING around the other block (any side will do)."""
        return abs(float(np.linalg.norm(block[:2] - base[:2])) - self.SPACING)

    def placed(self, facts: frozenset, block_name: str, base_name: str) -> bool:
        return Predicate("on-table", (block_name,)) in facts and Predicate("beside", (block_name, base_name)) in facts


@dataclass(frozen=True)
class FetchPutdownRewardConfig:
    # General costs
    time_penalty: float = 0.01
    action_l2_penalty: float = 0.001
    # Smoothness: penalize the change between consecutive actions, a proxy for jerk (the rate of change
    # of acceleration). The descent-speed cap below limits speed only, not abrupt changes.
    smoothness_penalty: float = 0.005

    # 1. Lower the block toward the table
    descend_progress_weight: float = 20.0
    descend_dense_weight: float = 0.10
    descend_dense_sharpness: float = 15.0
    max_height_progress: float = 0.025

    # 1b. Place it gently: penalize descending (or dropping) faster than this per-step speed, so the block
    #     is lowered under control rather than slammed or let fall (a full-speed slam is ~0.03 m/step)
    gentle_descent_speed: float = 0.012
    gentle_descent_penalty: float = 25.0

    # 2. Keep the block square (faces axis-aligned); squareness is in (0, 1], 1 for an un-rotated block
    square_progress_weight: float = 6.0
    square_sharpness: float = 6.0
    max_square_progress: float = 0.10
    square_land_bonus: float = 3.0

    # 3. Do not open the gripper while the block is still high (that drops it instead of placing it)
    early_open_height: float = 0.040
    early_open_penalty: float = 1.0

    # 4. Settle on the table
    on_table_bonus: float = 2.0

    # 5. Release once resting on the table
    release_progress_weight: float = 8.0
    release_bonus: float = 3.0

    # 6. Complete putdown; the bonus is scaled by squareness, from success_square_floor (tilted) to 1 (square)
    success_bonus: float = 15.0
    success_square_floor: float = 0.5


def squareness(rotation: np.ndarray | None, sharpness: float) -> float:
    """How axis-aligned a block is, in (0, 1]: each Euler angle's distance to the nearest multiple of 90
    degrees, summed, through exp(-sharpness * total). 1.0 for a flat, un-rotated block."""
    if rotation is None:
        return 1.0
    remainder = np.mod(np.asarray(rotation, dtype=np.float64), np.pi / 2)
    deviation = np.minimum(remainder, np.pi / 2 - remainder)
    return float(np.exp(-sharpness * float(np.sum(deviation))))


@dataclass
class FetchPutdownReward(RewardFunction):
    """Staged putdown shaping, starting with the block held (setup picks it up): lower it gently and square
    onto the table, without opening early, then release, success. Ported from
    deprecated/fetch_blockworld/rewards/putdown_rewards.py, plus a smoothness penalty on action changes.
    """

    config: FetchPutdownRewardConfig = field(default_factory=FetchPutdownRewardConfig)

    def __post_init__(self):
        self.reset(None, {})

    def reset(self, grounded_action_model, info) -> None:
        self.previous_height: float | None = None
        self.previous_squareness: float | None = None
        self.previous_gripper_width: float | None = None
        self.previous_action: np.ndarray | None = None
        self.square_land_earned = False
        self.on_table_earned = False
        self.release_earned = False
        self.success_earned = False

    def __call__(self, grounded_action_model, action, info, success):
        cfg = self.config
        state = info["environment_state"]
        block = grounded_action_model.arguments[0]
        height = max(0.0, float(state.block_positions[block][2] - TABLE_REST_Z))
        square = squareness(state.block_rotations.get(block), cfg.square_sharpness)
        gripper_width = float(state.finger_width)
        action = np.asarray(action, dtype=np.float64)
        on_table = Predicate("on-table", (block,)) in info["facts"]
        released = Predicate("gripper-empty", ()) in info["facts"]

        components = {
            "time": -cfg.time_penalty,
            "movement": -cfg.action_l2_penalty * float(np.dot(action, action)),
            "smoothness": 0.0,
            "descend": 0.0,
            "gentle": 0.0,
            "square": 0.0,
            "early_open": 0.0,
            "on_table": 0.0,
            "release": 0.0,
            "success": 0.0,
        }

        if self.previous_action is not None:
            change = action - self.previous_action
            components["smoothness"] = -cfg.smoothness_penalty * float(np.dot(change, change))

        # 1. Lower the block toward the table until it rests there.
        if not on_table:
            if self.previous_height is not None:
                progress = np.clip(self.previous_height - height, -cfg.max_height_progress, cfg.max_height_progress)
                components["descend"] += cfg.descend_progress_weight * float(progress)
            components["descend"] += cfg.descend_dense_weight * float(np.exp(-cfg.descend_dense_sharpness * height))

        # 1b. Penalize lowering or dropping the block faster than a gentle speed.
        if self.previous_height is not None:
            descent_speed = max(0.0, self.previous_height - height)
            components["gentle"] = -cfg.gentle_descent_penalty * max(0.0, descent_speed - cfg.gentle_descent_speed)

        # 2. Reward becoming more square, and a bonus scaled by squareness when it first comes to rest.
        if self.previous_squareness is not None:
            components["square"] += cfg.square_progress_weight * float(
                np.clip(square - self.previous_squareness, -cfg.max_square_progress, cfg.max_square_progress)
            )
        if on_table and not self.square_land_earned:
            components["square"] += cfg.square_land_bonus * square
            self.square_land_earned = True

        # 3. Penalize opening the gripper while the block is still too high.
        if gripper_width >= OPEN_FINGER_WIDTH and height > cfg.early_open_height:
            components["early_open"] = -cfg.early_open_penalty

        # 4. One-time reward for the block reaching the table.
        if on_table and not self.on_table_earned:
            components["on_table"] = cfg.on_table_bonus
            self.on_table_earned = True

        # 5. Reward opening the gripper, only once the block rests on the table.
        if on_table:
            if self.previous_gripper_width is not None:
                components["release"] = cfg.release_progress_weight * float(
                    np.clip(gripper_width - self.previous_gripper_width, -0.02, 0.02)
                )
            if released and not self.release_earned:
                components["release"] += cfg.release_bonus
                self.release_earned = True

        # 6. One-time terminal reward, worth more for a square placement.
        if success and not self.success_earned:
            scale = cfg.success_square_floor + (1.0 - cfg.success_square_floor) * square
            components["success"] = cfg.success_bonus * scale
            self.success_earned = True

        self.previous_height = height
        self.previous_squareness = square
        self.previous_gripper_width = gripper_width
        self.previous_action = action
        components = {name: float(value) for name, value in components.items()}
        return sum(components.values()), components


# Shaped reward function per skill; skills not listed use SparseReward.
FETCH_SHAPED_REWARDS = {
    "pickup": FetchPickupReward,
    "putdown": FetchPutdownReward,
    "stack": FetchStackReward,
    "unstack": FetchUnstackReward,
    "pickup-raised": FetchPickupRaisedReward,
    # unstack's default lift target, 0.06 above the block below (0.11 above the table), already reaches raised
    "unstack-raised": FetchUnstackReward,
    "place-beside": FetchPlaceBesideReward,
}
