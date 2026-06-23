from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PickupRewardConfig:
    # General costs
    time_penalty: float = 0.01
    action_l2_penalty: float = 0.001

    # 1. Approach object
    reach_progress_weight: float = 15.0
    reach_dense_weight: float = 0.10

    # 3. Get above object
    above_xy_tolerance: float = 0.030
    above_min_clearance: float = 0.040
    above_bonus: float = 1.5

    # 4. Lower into grasp pose
    grasp_xy_tolerance: float = 0.020
    grasp_z_offset: float = -0.005
    grasp_z_tolerance: float = 0.012
    descend_progress_weight: float = 20.0
    grasp_pose_bonus: float = 2.0

    # 5. Close/grip
    close_progress_weight: float = 10.0
    grip_bonus: float = 4.0

    # 6. Lift object
    off_table_height: float = 0.010
    off_table_bonus: float = 2.0
    lift_progress_weight: float = 8.0

    # 7. Complete pickup
    success_bonus: float = 15.0

    # Prevent single noisy transitions from producing huge rewards.
    max_position_progress: float = 0.025
    max_lift_fraction_progress: float = 0.25


@dataclass
class PickupRewardShaper:
    config: PickupRewardConfig = field(default_factory=PickupRewardConfig)

    previous_distance: float | None = None
    previous_grasp_error: float | None = None
    previous_gripper_width: float | None = None
    previous_lift_fraction: float = 0.0

    above_earned: bool = False
    grasp_pose_earned: bool = False
    grip_earned: bool = False
    off_table_earned: bool = False
    success_earned: bool = False

    def reset(self) -> None:
        self.previous_distance = None
        self.previous_grasp_error = None
        self.previous_gripper_width = None
        self.previous_lift_fraction = 0.0

        self.above_earned = False
        self.grasp_pose_earned = False
        self.grip_earned = False
        self.off_table_earned = False
        self.success_earned = False

    def compute(
        self,
        *,
        grip_pos: np.ndarray,
        object_pos: np.ndarray,
        gripper_width: float,
        lift: float,
        target_lift: float,
        action: np.ndarray,
        facts: dict[str, bool],
        success: bool,
    ) -> tuple[float, dict[str, float]]:
        cfg = self.config

        grip_pos = np.asarray(grip_pos, dtype=np.float64)
        object_pos = np.asarray(object_pos, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)

        distance = float(np.linalg.norm(grip_pos - object_pos))
        xy_distance = float(np.linalg.norm(grip_pos[:2] - object_pos[:2]))

        z_clearance = float(grip_pos[2] - object_pos[2])
        grasp_target_z = float(object_pos[2] + cfg.grasp_z_offset)
        grasp_z_error = abs(float(grip_pos[2] - grasp_target_z))

        # Includes XY alignment so moving sideways away from the block is penalized.
        grasp_error = float(
            np.sqrt(xy_distance**2 + grasp_z_error**2)
        )

        target_lift = max(float(target_lift), 1e-6)
        lift_fraction = float(np.clip(lift / target_lift, 0.0, 1.0))

        above_now = (
            xy_distance <= cfg.above_xy_tolerance
            and z_clearance >= cfg.above_min_clearance
        )

        at_grasp_pose = (
            xy_distance <= cfg.grasp_xy_tolerance
            and grasp_z_error <= cfg.grasp_z_tolerance
        )

        # Forgiving learning predicate:
        # either the reliable holding predicate is true, or the gripper is
        # closed while correctly positioned around the block.
        gripping_now = bool(
            facts.get("holding_object", False)
            or (
                at_grasp_pose
                and facts.get("gripper_closed", False)
            )
        )

        off_table_now = lift >= cfg.off_table_height

        components: dict[str, float] = {
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

        # 1. Approach the block.
        # Reward reduction in distance and provide a tiny absolute proximity
        # signal so the agent has guidance from the first step.
        if not self.above_earned:
            if self.previous_distance is not None:
                progress = np.clip(
                    self.previous_distance - distance,
                    -cfg.max_position_progress,
                    cfg.max_position_progress,
                )
                components["reach"] += cfg.reach_progress_weight * float(progress)

            components["reach"] += (
                cfg.reach_dense_weight * np.exp(-15.0 * distance)
            )

        # 3. One-time reward for becoming aligned over the block.
        if above_now and not self.above_earned:
            components["above"] = cfg.above_bonus
            self.above_earned = True

        # 4. Reward lowering toward the grasp pose.
        if self.above_earned and not self.grasp_pose_earned:
            if self.previous_grasp_error is not None:
                progress = np.clip(
                    self.previous_grasp_error - grasp_error,
                    -cfg.max_position_progress,
                    cfg.max_position_progress,
                )
                components["descend"] = (
                    cfg.descend_progress_weight * float(progress)
                )

            if at_grasp_pose:
                components["descend"] += cfg.grasp_pose_bonus
                self.grasp_pose_earned = True

        # 5. Reward closing only after reaching the grasp pose.
        if self.grasp_pose_earned and not self.grip_earned:
            if self.previous_gripper_width is not None:
                close_progress = np.clip(
                    self.previous_gripper_width - gripper_width,
                    -0.02,
                    0.02,
                )
                components["close"] = (
                    cfg.close_progress_weight * float(close_progress)
                )

            if gripping_now:
                components["grip"] = cfg.grip_bonus
                self.grip_earned = True

        # 6. Reward all upward object movement after gripping.
        if self.grip_earned or lift > 0.0:
            lift_progress = np.clip(
                lift_fraction - self.previous_lift_fraction,
                -cfg.max_lift_fraction_progress,
                cfg.max_lift_fraction_progress,
            )
            components["lift"] = (
                cfg.lift_progress_weight * float(lift_progress)
            )

        if off_table_now and not self.off_table_earned:
            components["off_table"] = cfg.off_table_bonus
            self.off_table_earned = True

        # 7. Large one-time terminal success reward.
        if success and not self.success_earned:
            components["success"] = cfg.success_bonus
            self.success_earned = True

        self.previous_distance = distance
        self.previous_grasp_error = grasp_error
        self.previous_gripper_width = float(gripper_width)
        self.previous_lift_fraction = lift_fraction

        return float(sum(components.values())), components