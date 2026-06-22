from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PutdownRewardConfig:
    # General costs
    time_penalty: float = 0.01
    action_l2_penalty: float = 0.001

    # 1. Lower the block toward the table
    descend_progress_weight: float = 20.0
    descend_dense_weight: float = 0.10
    descend_dense_sharpness: float = 15.0
    max_height_progress: float = 0.025

    # 2. Keep the block square (orientation aligned with the world/gripper axes).
    # squareness is in (0, 1]; 1.0 means every face is axis-aligned (a flat,
    # un-rotated block in line with the downward gripper).
    square_progress_weight: float = 6.0
    square_sharpness: float = 6.0
    max_square_progress: float = 0.10
    square_land_bonus: float = 3.0

    # 3. Do not drop the block before it is low enough.
    early_open_height: float = 0.040
    early_open_penalty: float = 1.0

    # 4. Settle on the table
    on_table_bonus: float = 2.0

    # 5. Release the block once it is resting on the table.
    # NOTE: a gripper clamped on the block reads a finger width of ~0.05, which
    # is wide enough to trip the generic "gripper_open" fact (threshold 0.035).
    # So "released" is judged here against a dedicated, much wider threshold
    # (a fully opened gripper reads ~0.10) instead of that fact.
    release_width: float = 0.075
    release_progress_weight: float = 8.0
    release_bonus: float = 3.0

    # 6. Complete putdown (block on the table AND gripper released).
    # The terminal bonus is scaled by squareness so a square placement pays more.
    success_bonus: float = 15.0
    success_square_floor: float = 0.5


def _squareness(object_rot: np.ndarray | None, sharpness: float) -> float:
    """Return how axis-aligned the block is, in (0, 1].

    Each Euler angle is measured against the nearest multiple of 90 degrees;
    a perfectly flat, un-rotated block scores 1.0 and a tilted/skewed one
    decays toward 0.
    """
    if object_rot is None:
        return 1.0
    rot = np.asarray(object_rot, dtype=np.float64)
    half = np.pi / 2.0
    mod = np.mod(rot, half)
    deviation = np.minimum(mod, half - mod)  # each in [0, pi/4]
    total = float(np.sum(deviation))
    return float(np.exp(-sharpness * total))


@dataclass
class PutdownRewardShaper:
    config: PutdownRewardConfig = field(default_factory=PutdownRewardConfig)

    previous_height: float | None = None
    previous_squareness: float | None = None
    previous_gripper_width: float | None = None

    square_land_earned: bool = False
    on_table_earned: bool = False
    release_earned: bool = False
    success_earned: bool = False

    def reset(self) -> None:
        self.previous_height = None
        self.previous_squareness = None
        self.previous_gripper_width = None

        self.square_land_earned = False
        self.on_table_earned = False
        self.release_earned = False
        self.success_earned = False

    def compute(
        self,
        *,
        object_rot: np.ndarray | None,
        gripper_width: float,
        height_above_table: float,
        action: np.ndarray,
        facts: dict[str, bool],
    ) -> tuple[float, dict[str, float], bool]:
        """Return reward, reward components, and the putdown success flag.

        A putdown is only complete once the block is resting on the table AND
        the gripper has been opened to release it, so success is computed here
        rather than from a single fact.
        """
        cfg = self.config

        action = np.asarray(action, dtype=np.float64)
        height = max(0.0, float(height_above_table))
        squareness = _squareness(object_rot, cfg.square_sharpness)

        on_table = bool(facts.get("object_on_table", False))
        released = bool(gripper_width >= cfg.release_width)

        components: dict[str, float] = {
            "time": -cfg.time_penalty,
            "movement": -cfg.action_l2_penalty * float(np.dot(action, action)),
            "descend": 0.0,
            "square": 0.0,
            "early_open": 0.0,
            "on_table": 0.0,
            "release": 0.0,
            "success": 0.0,
        }

        # 1. Lower the block toward the table until it is resting.
        if not on_table:
            if self.previous_height is not None:
                progress = np.clip(
                    self.previous_height - height,
                    -cfg.max_height_progress,
                    cfg.max_height_progress,
                )
                components["descend"] += cfg.descend_progress_weight * float(progress)

            components["descend"] += cfg.descend_dense_weight * float(
                np.exp(-cfg.descend_dense_sharpness * height)
            )

        # 2. Reward becoming more square, and a one-time bonus (scaled by how
        #    square it is) the moment the block first comes to rest.
        if self.previous_squareness is not None:
            square_progress = np.clip(
                squareness - self.previous_squareness,
                -cfg.max_square_progress,
                cfg.max_square_progress,
            )
            components["square"] += cfg.square_progress_weight * float(square_progress)

        if on_table and not self.square_land_earned:
            components["square"] += cfg.square_land_bonus * squareness
            self.square_land_earned = True

        # 3. Penalize releasing the block while it is still too high
        #    (which would drop it rather than place it).
        if released and height > cfg.early_open_height:
            components["early_open"] = -cfg.early_open_penalty

        # 4. One-time reward for the block reaching the table.
        if on_table and not self.on_table_earned:
            components["on_table"] = cfg.on_table_bonus
            self.on_table_earned = True

        # 5. Reward opening the gripper to release, only once on the table.
        if on_table:
            if self.previous_gripper_width is not None:
                open_progress = np.clip(
                    gripper_width - self.previous_gripper_width,
                    -0.02,
                    0.02,
                )
                components["release"] += cfg.release_progress_weight * float(open_progress)

            if released and not self.release_earned:
                components["release"] += cfg.release_bonus
                self.release_earned = True

        # 6. Large terminal reward once the block is placed and released,
        #    scaled by squareness so a square placement is worth more.
        success = bool(on_table and released)
        if success and not self.success_earned:
            square_scale = cfg.success_square_floor + (
                1.0 - cfg.success_square_floor
            ) * squareness
            components["success"] = cfg.success_bonus * square_scale
            self.success_earned = True

        self.previous_height = height
        self.previous_squareness = squareness
        self.previous_gripper_width = float(gripper_width)

        return float(sum(components.values())), components, success
