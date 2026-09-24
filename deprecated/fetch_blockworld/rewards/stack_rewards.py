"""Staged reward shaper for the ``stack`` skill.

Stack = pick up the designated *mover* block and place it on top of the
designated *base* block, then release. Mirrors the staged-milestone pattern of
pickup_rewards.py / putdown_rewards.py: approach -> grasp -> lift -> carry above
base -> gentle lower -> release. Success (computed here, like putdown) requires
the blocks to be stacked AND the gripper released.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class StackRewardConfig:
    # General costs
    time_penalty: float = 0.01
    action_l2_penalty: float = 0.001

    # 1. Approach the mover block
    reach_progress_weight: float = 15.0
    reach_dense_weight: float = 0.10
    reach_dense_sharpness: float = 15.0

    # 2. Grasp
    grasp_bonus: float = 4.0

    # 3. Lift the mover clear above the base so it can be carried
    carry_clearance: float = 0.06
    lift_progress_weight: float = 10.0
    lifted_bonus: float = 2.0

    # 4. Align horizontally over the base
    align_progress_weight: float = 12.0
    aligned_xy_tol: float = 0.030
    aligned_bonus: float = 2.0

    # 5. Lower gently onto the base
    descend_progress_weight: float = 18.0
    gentle_descent_speed: float = 0.012
    gentle_descent_penalty: float = 25.0
    stacked_bonus: float = 4.0

    # 6. Release (a gripper clamped on a block reads ~0.05, so use a wide
    #    threshold like putdown's).
    release_width: float = 0.075
    release_progress_weight: float = 8.0
    release_bonus: float = 3.0

    # 7. Completion
    success_bonus: float = 15.0

    # Safety clamps
    max_position_progress: float = 0.025
    object_half_size: float = 0.025


@dataclass
class StackRewardShaper:
    config: StackRewardConfig = field(default_factory=StackRewardConfig)

    previous_reach: float | None = None
    previous_lift_gap: float | None = None
    previous_align: float | None = None
    previous_stack_gap: float | None = None
    previous_mover_z: float | None = None
    previous_gripper_width: float | None = None

    grasped_earned: bool = False
    lifted_earned: bool = False
    aligned_earned: bool = False
    stacked_earned: bool = False
    release_earned: bool = False
    success_earned: bool = False

    def reset(self) -> None:
        self.previous_reach = None
        self.previous_lift_gap = None
        self.previous_align = None
        self.previous_stack_gap = None
        self.previous_mover_z = None
        self.previous_gripper_width = None
        self.grasped_earned = False
        self.lifted_earned = False
        self.aligned_earned = False
        self.stacked_earned = False
        self.release_earned = False
        self.success_earned = False

    def compute(
        self,
        *,
        grip_pos: np.ndarray,
        mover_pos: np.ndarray,
        base_pos: np.ndarray,
        gripper_width: float,
        action: np.ndarray,
        facts: dict[str, bool],
    ) -> tuple[float, dict[str, float], bool]:
        cfg = self.config
        grip = np.asarray(grip_pos, dtype=np.float64)
        mover = np.asarray(mover_pos, dtype=np.float64)
        base = np.asarray(base_pos, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)

        reach = float(np.linalg.norm(grip - mover))
        align = float(np.linalg.norm(mover[:2] - base[:2]))
        carry_target_z = float(base[2] + 2 * cfg.object_half_size + cfg.carry_clearance)
        stack_target_z = float(base[2] + 2 * cfg.object_half_size)
        lift_gap = max(0.0, carry_target_z - float(mover[2]))
        stack_gap = abs(float(mover[2]) - stack_target_z)

        gripping_now = bool(
            facts.get("gripper_closed", False) and facts.get("gripper_near_mover", False)
        )
        blocks_stacked = bool(facts.get("blocks_stacked", False))
        released = bool(gripper_width >= cfg.release_width)

        components: dict[str, float] = {
            "time": -cfg.time_penalty,
            "movement": -cfg.action_l2_penalty * float(np.dot(action, action)),
            "reach": 0.0,
            "grasp": 0.0,
            "lift": 0.0,
            "align": 0.0,
            "descend": 0.0,
            "gentle": 0.0,
            "stacked": 0.0,
            "release": 0.0,
            "success": 0.0,
        }

        # 1. Approach the mover until grasped.
        if not self.grasped_earned:
            if self.previous_reach is not None:
                progress = np.clip(
                    self.previous_reach - reach,
                    -cfg.max_position_progress,
                    cfg.max_position_progress,
                )
                components["reach"] += cfg.reach_progress_weight * float(progress)
            components["reach"] += cfg.reach_dense_weight * float(
                np.exp(-cfg.reach_dense_sharpness * reach)
            )
            if gripping_now:
                components["grasp"] = cfg.grasp_bonus
                self.grasped_earned = True

        # 2. Lift clear above the base.
        if self.grasped_earned and not self.lifted_earned:
            if self.previous_lift_gap is not None:
                progress = np.clip(
                    self.previous_lift_gap - lift_gap,
                    -cfg.max_position_progress,
                    cfg.max_position_progress,
                )
                components["lift"] += cfg.lift_progress_weight * float(progress)
            if lift_gap <= 1e-3:
                components["lift"] += cfg.lifted_bonus
                self.lifted_earned = True

        # 3. Align horizontally over the base.
        if self.lifted_earned and not self.aligned_earned:
            if self.previous_align is not None:
                progress = np.clip(
                    self.previous_align - align,
                    -cfg.max_position_progress,
                    cfg.max_position_progress,
                )
                components["align"] += cfg.align_progress_weight * float(progress)
            if align <= cfg.aligned_xy_tol:
                components["align"] += cfg.aligned_bonus
                self.aligned_earned = True

        # 4. Lower gently onto the base.
        if self.aligned_earned and not self.stacked_earned:
            if self.previous_stack_gap is not None:
                progress = np.clip(
                    self.previous_stack_gap - stack_gap,
                    -cfg.max_position_progress,
                    cfg.max_position_progress,
                )
                components["descend"] += cfg.descend_progress_weight * float(progress)
            if self.previous_mover_z is not None:
                descent_speed = max(0.0, self.previous_mover_z - float(mover[2]))
                excess = max(0.0, descent_speed - cfg.gentle_descent_speed)
                components["gentle"] = -cfg.gentle_descent_penalty * excess
            if blocks_stacked:
                components["stacked"] += cfg.stacked_bonus
                self.stacked_earned = True

        # 5. Release once stacked.
        if blocks_stacked:
            if self.previous_gripper_width is not None:
                open_progress = np.clip(
                    gripper_width - self.previous_gripper_width, -0.02, 0.02
                )
                components["release"] += cfg.release_progress_weight * float(open_progress)
            if released and not self.release_earned:
                components["release"] += cfg.release_bonus
                self.release_earned = True

        # 6. Completion: stacked and released.
        success = bool(blocks_stacked and released)
        if success and not self.success_earned:
            components["success"] = cfg.success_bonus
            self.success_earned = True

        self.previous_reach = reach
        self.previous_lift_gap = lift_gap
        self.previous_align = align
        self.previous_stack_gap = stack_gap
        self.previous_mover_z = float(mover[2])
        self.previous_gripper_width = float(gripper_width)

        return float(sum(components.values())), components, success
