"""Shaped Fetch rewards (ported from the original deprecated/fetch_blockworld/rewards/, removed; see git commit 59054395).

Each reads the moved block from the grounded action's first argument (stack: the
base block from the second) and the scene from info["environment_state"]
(FetchState) and info["facts"]. Every reward is staged: each stage has dense
progress terms (the change since the previous step) and a one-time bonus, and
every term is reported as a named reward component.

Progress terms are not clipped per step, so over an episode each sums to the net
progress, and moving back and forth earns nothing. Clipped, a fast move back would
be charged less than the slow move forward earned: holding a block and lifting it
slowly, then lowering it fast, would earn about 25 from pickup's lift term per
episode against an honest 8, as much as a success, without ever succeeding.

A failure (the moved block dropped, another block knocked aside, the gripper pressing hard) ends the episode on the
step it happens, before its consequences play out: a block let go 20 cm up has not fallen yet. So on a failing step
every reward charges back the moved block's progress of the episode (the refund component), as the block's fall
would have; the stage bonuses and the gripper's approach are kept, since failing cannot earn them. Without the
refund, lifting a block and letting it go kept the lift's reward: with a small failure penalty it paid about 10 per
episode, and a policy learned it instead of the skill (pickup-raised trained alone: 3 in 4 training episodes failing
and none succeeding at 60k steps).
"""

from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from cam.domain.symbols import Predicate
from cam.environments.fetch.fetch_multiblock_environment import BLOCK_HALF_SIZE
from cam.environments.fetch.fetch_predicate_evaluation_wrapper import (
    SETTLE_DISTANCE,
    BESIDE_DISTANCE,
    OPEN_FINGER_WIDTH,
    MAX_GRIPPER_FORCE,
    SAFE_GRIPPER_FORCE,
    RAISED_HEIGHT,
    TABLE_REST_Z,
    gripping,
    push_goal,
)
from cam.rewards.reward_function import RewardFunction



# Subtracted when an episode fails (a block dropped or knocked aside, the gripper pressing hard), besides the refund of
# the block's progress; success pays 15. Small on purpose: an exploring policy fails often near the blocks (a noisy
# finger command lets a lifted block go; a noisy height command presses the fingers into the table), and at 10
# policies learned to keep away from them (hovering over the block with the fingers closed, never grasping). It only
# has to make failing worse than waiting: the time penalty costs at most 1 from any step on (0.05 per step,
# discounted at gamma = 0.95), and a failure also forfeits the success to come.
FAILURE_PENALTY = 2.0


def refund(info: dict, block_progress: float) -> float:
    """The refund component: on a step that ends the episode as a failure (info["failed"]), minus the moved block's
    progress earned in the episode (see the module docstring); otherwise 0."""
    return -block_progress if info.get("failed") else 0.0


@dataclass(frozen=True)
class FetchMotionRewardConfig:
    # Light on purpose: these terms grow with exploration noise, wherever the gripper touches something, so heavier
    # weights teach an exploring policy to keep away from the blocks before it ever finds a success.
    smoothness_penalty: float = 0.0005  # per unit squared of change between consecutive actions
    acceleration_penalty: float = 50.0  # per (m/step^2)^2 of the moved block's acceleration
    disturbance_penalty: float = 2.0  # per metre the other blocks move
    # per newton of net gripper contact force from SAFE_ to MAX_GRIPPER_FORCE: at most 0.2 per step, so a press that
    # lasts until it fails (FORCE_FAILURE_STEPS) costs half as much as the failure itself
    force_penalty: float = 0.0005


@dataclass
class FetchMotionReward(RewardFunction):
    """Reward terms added to every skill's reward (SkillEnvironment's shared_reward), against brutal motion:
        smoothness    the change between consecutive actions, squared
        acceleration  the moved block's (first argument) acceleration (second difference of its position over
                      steps), squared: high when it is yanked, thrown or slammed
        disturbance   how far the other blocks moved this step: any knock is penalized (moving one more than
                      DISTURB_DISTANCE from its start also ends the episode as a failure)
        force         net contact force on the gripper above SAFE_GRIPPER_FORCE (100 N), counted up to
                      MAX_GRIPPER_FORCE (500 N): at most 0.2 per step. One-step contact spikes reach tens of kN
                      in simulation; uncapped, one would cost tens of points, more than success earns, and
                      exploring would score worse than doing nothing. Above MAX_GRIPPER_FORCE for
                      FORCE_FAILURE_STEPS consecutive steps the episode fails. Required for real robots.
    A full-speed scripted episode costs about 0.005-0.13; throwing a block costs some tenths of a point per step.
    """

    config: FetchMotionRewardConfig = field(default_factory=FetchMotionRewardConfig)

    def __post_init__(self):
        self.reset(None, {})

    def reset(self, grounded_action_model, info) -> None:
        self.previous_action: np.ndarray | None = None
        self.moved_positions: deque = deque(maxlen=3)
        self.previous_others: dict[str, np.ndarray] = {}

    def __call__(self, grounded_action_model, action, info, success):
        cfg = self.config
        positions = info["environment_state"].block_positions
        moved = grounded_action_model.arguments[0]
        action = np.asarray(action, dtype=np.float64)
        smoothness = 0.0 if self.previous_action is None else float(np.sum((action - self.previous_action) ** 2))
        self.previous_action = action
        self.moved_positions.append(np.asarray(positions[moved], dtype=np.float64))
        acceleration = 0.0
        if len(self.moved_positions) == 3:
            p0, p1, p2 = self.moved_positions
            acceleration = float(np.sum((p2 - 2 * p1 + p0) ** 2))
        others = {name: np.asarray(p, dtype=np.float64) for name, p in positions.items() if name != moved}
        disturbance = sum(float(np.linalg.norm(others[name] - self.previous_others[name]))
                          for name in others if name in self.previous_others)
        self.previous_others = others
        components = {
            "smoothness": -cfg.smoothness_penalty * smoothness,
            "acceleration": -cfg.acceleration_penalty * acceleration,
            "disturbance": -cfg.disturbance_penalty * disturbance,
            "force": -cfg.force_penalty * min(
                max(0.0, float(np.linalg.norm(info["environment_state"].gripper_force)) - SAFE_GRIPPER_FORCE),
                MAX_GRIPPER_FORCE - SAFE_GRIPPER_FORCE,
            ),
        }
        return sum(components.values()), components


@dataclass(frozen=True)
class FetchPickupRewardConfig:
    # General costs
    time_penalty: float = 0.05  # per step: at most 1 from any step on (gamma = 0.95), half a failure (FAILURE_PENALTY)
    action_l2_penalty: float = 0.001

    # 1. Approach object
    reach_progress_weight: float = 15.0

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
    # False: the lift term rewards rising toward target_lift and nothing above it (pickup and unstack may stop
    # anywhere below the raised band). True: it peaks at target_lift and falls on both sides, so overshooting costs
    # what rising earned (the raised variants, which must stop in the band)
    two_sided_lift: bool = False
    lift_off_height: float = 0.010
    lift_off_bonus: float = 2.0
    lift_progress_weight: float = 8.0

    # 6. Settle: once, the first step the held block has settled where the skill wants it (in the raised band for the
    #    raised variants, below it for pickup and unstack): risen or sunk less than SETTLE_DISTANCE over its last 3
    #    steps, so stopping there is rewarded before success (5 steps)
    settle_bonus: float = 3.0

    # 7. Complete pickup
    success_bonus: float = 15.0



@dataclass
class FetchPickupReward(RewardFunction):
    """Staged pickup shaping: approach, get above, descend, close, lift, settle, success.

    Progress terms reward the change since the previous step; each stage's bonus
    is paid once per episode. The lift is measured from rest_height (the table).
    An episode that starts with the open gripper already around the block (unstack's setup leaves it there, after
    placing the block) counts the approach stages as done, without their bonuses, so closing is rewarded from its
    first step: otherwise only getting 4 cm above the block first would open the close and grip stages.
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
        self.settle_earned = False
        self.success_earned = False
        self.recent_object_positions: deque = deque(maxlen=3)  # the settle bonus's window
        self.block_progress = 0.0  # the lift terms so far, refunded on failure
        if grounded_action_model is not None and "environment_state" in info:
            state = info["environment_state"]
            around = gripping(replace(state, finger_width=0.0), grounded_action_model.arguments[0])  # within reach
            self.above_earned = self.grasp_pose_earned = around

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
        if cfg.two_sided_lift:  # 1 at the target height, less above and below it
            lift_fraction = 1.0 - abs(lift - cfg.target_lift) / max(cfg.target_lift, 1e-6)
        else:
            lift_fraction = float(np.clip(lift / max(cfg.target_lift, 1e-6), 0.0, 1.0))

        above_now = xy_distance <= cfg.above_xy_tolerance and z_clearance >= cfg.above_min_clearance
        at_grasp_pose = xy_distance <= cfg.grasp_xy_tolerance and grasp_z_error <= cfg.grasp_z_tolerance
        # Forgiving: holding, or the closed fingers around the block (the predicates' gripping), lifted or not.
        gripping_now = Predicate("holding", (block,)) in info["facts"] or gripping(state, block)
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
            "settle": 0.0,
            "success": 0.0,
            "refund": 0.0,
        }

        # 1. Approach the block: reward reduced distance, in every stage, so moving away again (e.g. once the stage
        #    bonuses are earned) costs what approaching earned. Being near the block earns nothing by itself.
        if self.previous_distance is not None:
            progress = self.previous_distance - distance
            components["reach"] = cfg.reach_progress_weight * float(progress)

        # 2. One-time reward for becoming aligned over the block.
        if above_now and not self.above_earned:
            components["above"] = cfg.above_bonus
            self.above_earned = True

        # 3. Reward lowering toward the grasp pose.
        if self.above_earned and not self.grasp_pose_earned:
            if self.previous_grasp_error is not None:
                progress = self.previous_grasp_error - grasp_error
                components["descend"] = cfg.descend_progress_weight * float(progress)
            if at_grasp_pose:
                components["descend"] += cfg.grasp_pose_bonus
                self.grasp_pose_earned = True

        # 4. Reward closing only after reaching the grasp pose.
        if self.grasp_pose_earned and not self.grip_earned:
            if self.previous_gripper_width is not None:
                close_progress = self.previous_gripper_width - gripper_width
                components["close"] = cfg.close_progress_weight * float(close_progress)
            if gripping_now:
                components["grip"] = cfg.grip_bonus
                self.grip_earned = True

        # 5. Reward upward block movement after gripping.
        if self.grip_earned or lift > 0.0:
            lift_progress = lift_fraction - self.previous_lift_fraction
            components["lift"] = cfg.lift_progress_weight * float(lift_progress)
            self.block_progress += components["lift"]
        if lift_off_now and not self.lift_off_earned:
            components["lift_off"] = cfg.lift_off_bonus
            self.lift_off_earned = True

        # 6. One-time reward for stopping the held block where the skill wants it.
        held = Predicate("holding", (block,)) in info["facts"]
        in_place = (Predicate("raised", (block,)) in info["facts"]) == cfg.two_sided_lift
        self.recent_object_positions.append(object_pos)
        at_rest = (len(self.recent_object_positions) == self.recent_object_positions.maxlen
                   and abs(self.recent_object_positions[-1][2] - self.recent_object_positions[0][2]) < SETTLE_DISTANCE)
        if held and in_place and at_rest and not self.settle_earned:
            components["settle"] = cfg.settle_bonus
            self.settle_earned = True

        # 7. One-time terminal success reward.
        if success and not self.success_earned:
            components["success"] = cfg.success_bonus
            self.success_earned = True
        components["refund"] = refund(info, self.block_progress)

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
    time_penalty: float = 0.05  # per step: at most 1 from any step on (gamma = 0.95), half a failure (FAILURE_PENALTY)
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



@dataclass
class FetchStackReward(RewardFunction):
    """Staged stack shaping, starting with the block held (setup picks it up): carry it above the base,
    align over the base, lower gently onto it, release, success. Ported from
    the original deprecated/fetch_blockworld/rewards/stack_rewards.py without its approach and grasp stages, which
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
        self.block_progress = 0.0  # the carry, align and descend progress so far, refunded on failure

    def resting_z(self, block: np.ndarray, base: np.ndarray) -> float:
        """Height of the block's centre once placed: on top of the base."""
        return float(base[2] + 2 * BLOCK_HALF_SIZE)

    def align_distance(self, block: np.ndarray, base: np.ndarray) -> float:
        """Horizontal distance from the block to where it should be placed: over the base."""
        return float(np.linalg.norm(block[:2] - base[:2]))

    def placed(self, facts: frozenset, block_name: str, base_name: str) -> bool:
        return Predicate("on", (block_name, base_name)) in facts

    def progress(self, previous: float | None, current: float, weight: float) -> float:
        """weight x the reduction of a gap since the previous step."""
        if previous is None:
            return 0.0
        return weight * float(previous - current)

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
            "refund": 0.0,
        }

        # 1. Raise the held block high enough to clear the base.
        if held and not self.carried_earned:
            components["carry"] = self.progress(self.previous_carry_gap, carry_gap, cfg.carry_progress_weight)
            self.block_progress += components["carry"]
            if carry_gap <= cfg.carry_tolerance:
                components["carry"] += cfg.carried_bonus
                self.carried_earned = True

        # 2. Move the held block over the base.
        if held and self.carried_earned and not self.aligned_earned:
            components["align"] = self.progress(
                self.previous_align_distance, align_distance, cfg.align_progress_weight
            )
            self.block_progress += components["align"]
            if align_distance <= cfg.aligned_xy_tolerance:
                components["align"] += cfg.aligned_bonus
                self.aligned_earned = True

        # 3. Lower onto the base, penalizing a fast drop.
        if self.aligned_earned and not self.placed_earned:
            components["descend"] = self.progress(self.previous_place_gap, place_gap, cfg.descend_progress_weight)
            self.block_progress += components["descend"]
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
                    (gripper_width - self.previous_gripper_width)
                )
            if released and not self.release_earned:
                components["release"] += cfg.release_bonus
                self.release_earned = True

        # 5. One-time terminal success reward.
        if success and not self.success_earned:
            components["success"] = cfg.success_bonus
            self.success_earned = True
        components["refund"] = refund(info, self.block_progress)

        self.previous_carry_gap = carry_gap
        self.previous_align_distance = align_distance
        self.previous_place_gap = place_gap
        self.previous_block_z = float(block[2])
        self.previous_gripper_width = gripper_width
        components = {name: float(value) for name, value in components.items()}
        return sum(components.values()), components


@dataclass
class FetchPickupRaisedReward(FetchPickupReward):
    """pickup's stages with the lift target at the middle of the raised band above the table, so the lift stage leads
    to (raised ?o); the lift term is two-sided, so overshooting the band costs what rising into it earned, and the
    settle bonus is paid at rest in the band."""

    config: FetchPickupRewardConfig = field(
        default_factory=lambda: FetchPickupRewardConfig(target_lift=float(np.mean(RAISED_HEIGHT)), two_sided_lift=True)
    )


@dataclass
class FetchUnstackRaisedReward(FetchUnstackReward):
    """unstack's stages with the lift target at the middle of the raised band; the lift is measured from where
    the block rests on the block below, one block height above the table. Two-sided, with the settle bonus in the
    band, as for pickup-raised."""

    config: FetchPickupRewardConfig = field(
        default_factory=lambda: FetchPickupRewardConfig(target_lift=float(np.mean(RAISED_HEIGHT)) - 2 * BLOCK_HALF_SIZE,
                                                        two_sided_lift=True)
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
    time_penalty: float = 0.05  # per step: at most 1 from any step on (gamma = 0.95), half a failure (FAILURE_PENALTY)
    action_l2_penalty: float = 0.001

    # 1. Lower the block toward the table
    descend_progress_weight: float = 20.0
    descend_dense_weight: float = 0.10
    descend_dense_sharpness: float = 15.0

    # 1b. Place it gently: penalize descending (or dropping) faster than this per-step speed, so the block
    #     is lowered under control rather than slammed or let fall (a full-speed slam is ~0.03 m/step)
    gentle_descent_speed: float = 0.012
    gentle_descent_penalty: float = 25.0

    # 2. Keep the block square (faces axis-aligned); squareness is in (0, 1], 1 for an un-rotated block
    square_progress_weight: float = 6.0
    square_sharpness: float = 6.0
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
    the original deprecated/fetch_blockworld/rewards/putdown_rewards.py, plus a smoothness penalty on action changes.
    """

    config: FetchPutdownRewardConfig = field(default_factory=FetchPutdownRewardConfig)

    def __post_init__(self):
        self.reset(None, {})

    def reset(self, grounded_action_model, info) -> None:
        self.previous_height: float | None = None
        self.previous_squareness: float | None = None
        self.previous_gripper_width: float | None = None
        self.square_land_earned = False
        self.on_table_earned = False
        self.release_earned = False
        self.success_earned = False
        self.block_progress = 0.0  # the descend and square progress so far, refunded on failure

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
            "descend": 0.0,
            "gentle": 0.0,
            "square": 0.0,
            "early_open": 0.0,
            "on_table": 0.0,
            "release": 0.0,
            "success": 0.0,
            "refund": 0.0,
        }


        # 1. Lower the block toward the table until it rests there.
        if not on_table:
            if self.previous_height is not None:
                progress = cfg.descend_progress_weight * float(self.previous_height - height)
                components["descend"] += progress
                self.block_progress += progress
            components["descend"] += cfg.descend_dense_weight * float(np.exp(-cfg.descend_dense_sharpness * height))

        # 1b. Penalize lowering or dropping the block faster than a gentle speed.
        if self.previous_height is not None:
            descent_speed = max(0.0, self.previous_height - height)
            components["gentle"] = -cfg.gentle_descent_penalty * max(0.0, descent_speed - cfg.gentle_descent_speed)

        # 2. Reward becoming more square, and a bonus scaled by squareness when it first comes to rest.
        if self.previous_squareness is not None:
            progress = cfg.square_progress_weight * float(square - self.previous_squareness)
            components["square"] += progress
            self.block_progress += progress
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
                    (gripper_width - self.previous_gripper_width)
                )
            if released and not self.release_earned:
                components["release"] += cfg.release_bonus
                self.release_earned = True

        # 6. One-time terminal reward, worth more for a square placement.
        if success and not self.success_earned:
            scale = cfg.success_square_floor + (1.0 - cfg.success_square_floor) * square
            components["success"] = cfg.success_bonus * scale
            self.success_earned = True
        components["refund"] = refund(info, self.block_progress)

        self.previous_height = height
        self.previous_squareness = square
        self.previous_gripper_width = gripper_width
        components = {name: float(value) for name, value in components.items()}
        return sum(components.values()), components


@dataclass(frozen=True)
class FetchPushRewardConfig:
    time_penalty: float = 0.05  # per step: at most 1 from any step on (gamma = 0.95), half a failure (FAILURE_PENALTY)
    action_l2_penalty: float = 0.001
    # 1. Get behind the block (opposite its goal), at its height, by way of a point above it: from the goal side of
    #    the block (most starts) the straight line to the point behind it runs through the block
    behind_offset: float = 0.055  # gripper centre behind the block centre
    hover_height: float = 0.08  # the point above, over the block's centre height (as the scripted push)
    above_tolerance: float = 0.02  # horizontally, and how far below hover_height still counts as above
    above_bonus: float = 1.0
    reach_progress_weight: float = 15.0
    behind_tolerance: float = 0.02
    behind_bonus: float = 2.0
    # 2. Push the block toward its goal
    push_progress_weight: float = 30.0
    # 3. Complete push
    success_bonus: float = 15.0


@dataclass
class FetchPushReward(RewardFunction):
    """Staged push shaping: get above the point behind the block (opposite its goal, push_goal), lower to it,
    then push the block toward its goal, success. The approach goes over the block, as the scripted push does: the
    gripper usually starts on the goal side of the block, where heading straight for the point behind it drives into
    the block (pushing it the wrong way, and failing on force in about a third of such starts). Reaching the point
    behind directly still earns its bonus. The push progress counts at any stage, so any way of moving the block
    toward its goal is rewarded."""

    config: FetchPushRewardConfig = field(default_factory=FetchPushRewardConfig)

    def __post_init__(self):
        self.reset(None, {})

    def reset(self, grounded_action_model, info) -> None:
        self.previous_reach: float | None = None
        self.previous_goal_distance: float | None = None
        self.above_earned = False
        self.behind_earned = False
        self.success_earned = False
        self.block_progress = 0.0  # the push progress so far, refunded on failure

    def __call__(self, grounded_action_model, action, info, success):
        cfg = self.config
        state = info["environment_state"]
        block = np.asarray(state.block_positions[grounded_action_model.arguments[0]], dtype=np.float64)
        gripper = np.asarray(state.gripper_position, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        goal = push_goal(state, grounded_action_model)
        to_goal = goal - block[:2]
        goal_distance = float(np.linalg.norm(to_goal))
        direction = to_goal / max(goal_distance, 1e-6)
        behind = np.array([*(block[:2] - cfg.behind_offset * direction), block[2]])
        above = behind + [0.0, 0.0, cfg.hover_height]

        components = {
            "time": -cfg.time_penalty,
            "movement": -cfg.action_l2_penalty * float(np.dot(action, action)),
            "reach": 0.0,
            "above": 0.0,
            "behind": 0.0,
            "push": 0.0,
            "success": 0.0,
            "refund": 0.0,
        }
        # 1. Get above the point behind the block, then lower to it (directly to it also earns the behind bonus).
        reach = float(np.linalg.norm(gripper - (behind if self.above_earned else above)))
        if not self.behind_earned:
            if self.previous_reach is not None:
                components["reach"] = cfg.reach_progress_weight * float(self.previous_reach - reach)
            at_behind = float(np.linalg.norm(gripper - behind)) <= cfg.behind_tolerance
            if not self.above_earned and (at_behind or (
                    float(np.linalg.norm(gripper[:2] - behind[:2])) <= cfg.above_tolerance
                    and gripper[2] >= above[2] - cfg.above_tolerance)):
                components["above"] = cfg.above_bonus
                self.above_earned = True
                reach = float(np.linalg.norm(gripper - behind))  # the next leg's distance, so progress continues from it
            if at_behind:
                components["behind"] = cfg.behind_bonus
                self.behind_earned = True
        # 2. Move the block toward its goal.
        if self.previous_goal_distance is not None:
            components["push"] = cfg.push_progress_weight * float(
                self.previous_goal_distance - goal_distance
            )
            self.block_progress += components["push"]
        # 3. One-time terminal success reward.
        if success and not self.success_earned:
            components["success"] = cfg.success_bonus
            self.success_earned = True
        components["refund"] = refund(info, self.block_progress)

        self.previous_reach = reach
        self.previous_goal_distance = goal_distance
        components = {name: float(value) for name, value in components.items()}
        return sum(components.values()), components


# Shaped reward function per skill; skills not listed use SparseReward.
FETCH_SHAPED_REWARDS = {
    "pickup": FetchPickupReward,
    "putdown": FetchPutdownReward,
    "stack": FetchStackReward,
    "unstack": FetchUnstackReward,
    "pickup-raised": FetchPickupRaisedReward,
    "unstack-raised": FetchUnstackRaisedReward,
    "place-beside": FetchPlaceBesideReward,
    **{name: FetchPushReward for name in
       ("push-left", "push-right", "push-front", "push-back", "push-left-front", "push-right-front",
        "push-left-back", "push-right-back", "push-beside")},
}
