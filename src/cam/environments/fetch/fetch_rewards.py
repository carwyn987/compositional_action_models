"""Shaped Fetch rewards (ported from deprecated/fetch_blockworld/rewards/).

Each reads the target block from the grounded action's first argument and the
scene from info["environment_state"] (FetchState) and info["facts"].
"""

from dataclasses import dataclass, field

import numpy as np

from cam.domain.symbols import Predicate
from cam.environments.fetch.fetch_predicate_evaluation_wrapper import OPEN_FINGER_WIDTH, TABLE_REST_Z
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

    # 5. Lift object
    target_lift: float = 0.06
    off_table_height: float = 0.010
    off_table_bonus: float = 2.0
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
    is paid once per episode.
    """

    config: FetchPickupRewardConfig = field(default_factory=FetchPickupRewardConfig)

    def __post_init__(self):
        self.reset(None, {})

    def reset(self, grounded_action_model, info) -> None:
        self.previous_distance: float | None = None
        self.previous_grasp_error: float | None = None
        self.previous_gripper_width: float | None = None
        self.previous_lift_fraction = 0.0
        self.above_earned = False
        self.grasp_pose_earned = False
        self.grip_earned = False
        self.off_table_earned = False
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
        lift = max(0.0, float(object_pos[2] - TABLE_REST_Z))
        lift_fraction = float(np.clip(lift / max(cfg.target_lift, 1e-6), 0.0, 1.0))

        above_now = xy_distance <= cfg.above_xy_tolerance and z_clearance >= cfg.above_min_clearance
        at_grasp_pose = xy_distance <= cfg.grasp_xy_tolerance and grasp_z_error <= cfg.grasp_z_tolerance
        # Forgiving: holding, or the gripper closing while correctly positioned around the block.
        gripping_now = Predicate("holding", (block,)) in info["facts"] or (
            at_grasp_pose and gripper_width < OPEN_FINGER_WIDTH
        )
        off_table_now = lift >= cfg.off_table_height

        components = {
            "time": -cfg.time_penalty,
            "movement": -cfg.action_l2_penalty * float(np.dot(action, action)),
            "reach": 0.0,
            "above": 0.0,
            "descend": 0.0,
            "close": 0.0,
            "grip": 0.0,
            "off_table": 0.0,
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
        if off_table_now and not self.off_table_earned:
            components["off_table"] = cfg.off_table_bonus
            self.off_table_earned = True

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


# Shaped reward function per skill; skills not listed use SparseReward.
FETCH_SHAPED_REWARDS = {"pickup": FetchPickupReward}
