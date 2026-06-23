"""Staged reward shaper for the ``unstack`` skill.

Unstack = the *mover* block begins stacked on the *base* block; pick it off,
carry it clear of the base, set it gently on the table, and release. Mirrors the
staged-milestone pattern of the other shapers: approach -> grasp -> lift off ->
move clear -> gentle lower to table -> release. Success (computed here) requires
the mover resting on the table, clear of the base, no longer stacked, and the
gripper released.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class UnstackRewardConfig:
    # General costs
    time_penalty: float = 0.01
    action_l2_penalty: float = 0.001

    # 1. Approach the (stacked) mover block
    reach_progress_weight: float = 15.0
    reach_dense_weight: float = 0.10
    reach_dense_sharpness: float = 15.0

    # 2. Grasp
    grasp_bonus: float = 4.0

    # 3. Lift the mover clear off the base
    lift_clearance: float = 0.06
    lift_progress_weight: float = 10.0
    lifted_bonus: float = 2.0

    # 4. Move horizontally clear of the base
    clear_progress_weight: float = 12.0
    cleared_bonus: float = 2.0

    # 5. Lower gently to the table
    descend_progress_weight: float = 18.0
    descend_dense_weight: float = 0.10
    descend_dense_sharpness: float = 15.0
    gentle_descent_speed: float = 0.012
    gentle_descent_penalty: float = 25.0
    on_table_bonus: float = 4.0

    # 6. Release
    release_width: float = 0.075
    release_progress_weight: float = 8.0
    release_bonus: float = 3.0

    # 7. Completion
    success_bonus: float = 15.0

    # Safety clamps
    max_position_progress: float = 0.025
    object_half_size: float = 0.025


@dataclass
class UnstackRewardShaper:
    config: UnstackRewardConfig = field(default_factory=UnstackRewardConfig)

    previous_reach: float | None = None
    previous_lift_gap: float | None = None
    previous_clear: float | None = None
    previous_height: float | None = None
    previous_mover_z: float | None = None
    previous_gripper_width: float | None = None

    grasped_earned: bool = False
    lifted_earned: bool = False
    cleared_earned: bool = False
    on_table_earned: bool = False
    release_earned: bool = False
    success_earned: bool = False

    def reset(self) -> None:
        self.previous_reach = None
        self.previous_lift_gap = None
        self.previous_clear = None
        self.previous_height = None
        self.previous_mover_z = None
        self.previous_gripper_width = None
        self.grasped_earned = False
        self.lifted_earned = False
        self.cleared_earned = False
        self.on_table_earned = False
        self.release_earned = False
        self.success_earned = False

    def compute(
        self,
        *,
        grip_pos: np.ndarray,
        mover_pos: np.ndarray,
        base_pos: np.ndarray,
        gripper_width: float,
        table_z: float,
        clear_dist: float,
        action: np.ndarray,
        facts: dict[str, bool],
    ) -> tuple[float, dict[str, float], bool]:
        cfg = self.config
        grip = np.asarray(grip_pos, dtype=np.float64)
        mover = np.asarray(mover_pos, dtype=np.float64)
        base = np.asarray(base_pos, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)

        reach = float(np.linalg.norm(grip - mover))
        horiz = float(np.linalg.norm(mover[:2] - base[:2]))
        lift_target_z = float(base[2] + 2 * cfg.object_half_size + cfg.lift_clearance)
        lift_gap = max(0.0, lift_target_z - float(mover[2]))
        height = max(0.0, float(mover[2]) - float(table_z))

        gripping_now = bool(
            facts.get("gripper_closed", False) and facts.get("gripper_near_mover", False)
        )
        cleared = bool(horiz > clear_dist)
        on_table = bool(facts.get("mover_on_table", False))
        not_stacked = not bool(facts.get("blocks_stacked", False))
        released = bool(gripper_width >= cfg.release_width)

        components: dict[str, float] = {
            "time": -cfg.time_penalty,
            "movement": -cfg.action_l2_penalty * float(np.dot(action, action)),
            "reach": 0.0,
            "grasp": 0.0,
            "lift": 0.0,
            "clear": 0.0,
            "descend": 0.0,
            "gentle": 0.0,
            "on_table": 0.0,
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

        # 2. Lift the mover clear off the base.
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

        # 3. Move horizontally clear of the base.
        if self.lifted_earned and not self.cleared_earned:
            if self.previous_clear is not None:
                progress = np.clip(
                    horiz - self.previous_clear,
                    -cfg.max_position_progress,
                    cfg.max_position_progress,
                )
                components["clear"] += cfg.clear_progress_weight * float(progress)
            if cleared:
                components["clear"] += cfg.cleared_bonus
                self.cleared_earned = True

        # 4. Lower gently to the table once clear of the base.
        if self.cleared_earned and not self.on_table_earned:
            if self.previous_height is not None:
                progress = np.clip(
                    self.previous_height - height,
                    -cfg.max_position_progress,
                    cfg.max_position_progress,
                )
                components["descend"] += cfg.descend_progress_weight * float(progress)
            components["descend"] += cfg.descend_dense_weight * float(
                np.exp(-cfg.descend_dense_sharpness * height)
            )
            if self.previous_mover_z is not None:
                descent_speed = max(0.0, self.previous_mover_z - float(mover[2]))
                excess = max(0.0, descent_speed - cfg.gentle_descent_speed)
                components["gentle"] = -cfg.gentle_descent_penalty * excess
            if on_table:
                components["on_table"] += cfg.on_table_bonus
                self.on_table_earned = True

        # 5. Release once resting on the table, clear of the base.
        if on_table and cleared:
            if self.previous_gripper_width is not None:
                open_progress = np.clip(
                    gripper_width - self.previous_gripper_width, -0.02, 0.02
                )
                components["release"] += cfg.release_progress_weight * float(open_progress)
            if released and not self.release_earned:
                components["release"] += cfg.release_bonus
                self.release_earned = True

        # 6. Completion: mover on the table, clear of the base, unstacked, released.
        success = bool(on_table and cleared and not_stacked and released)
        if success and not self.success_earned:
            components["success"] = cfg.success_bonus
            self.success_earned = True

        self.previous_reach = reach
        self.previous_lift_gap = lift_gap
        self.previous_clear = horiz
        self.previous_height = height
        self.previous_mover_z = float(mover[2])
        self.previous_gripper_width = float(gripper_width)

        return float(sum(components.values())), components, success
