"""Shaped rewards for the directional push skills.

A single direction-parameterized shaper covers pushleft/right/forward/backward
so there is no per-direction duplicated reward code. Each push skill gets its
own shaper instance (with its own direction and progress/rotation state), so
the four skills are trained completely independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# 2D (x, y) unit vectors for each push skill. See the coordinate-convention note
# in facts.py: x-/x+ and y+/y-.
PUSH_DIRECTIONS: dict[str, np.ndarray] = {
    "pushleft": np.array([-1.0, 0.0]),
    "pushright": np.array([1.0, 0.0]),
    "pushforward": np.array([0.0, 1.0]),
    "pushbackward": np.array([0.0, -1.0]),
}


def require_push_direction(skill_name: str) -> np.ndarray:
    try:
        return PUSH_DIRECTIONS[skill_name].astype(np.float64)
    except KeyError as exc:
        raise KeyError(
            f"Unknown push skill {skill_name!r}. Choices: {sorted(PUSH_DIRECTIONS)}"
        ) from exc


@dataclass(frozen=True)
class PushRewardConfig:
    # General costs
    time_penalty: float = 0.01
    action_l2_penalty: float = 0.001

    # Progress along the target push direction (rewarded as per-step delta so it
    # cannot be farmed by stalling on an already-displaced block).
    progress_weight: float = 10.0
    max_progress: float = 0.025

    # Keep the block on the straight line along the push direction: penalize any
    # displacement perpendicular to it.
    lateral_penalty: float = 2.0

    # Keep the gripper close to the block so it stays in contact / behind it.
    proximity_weight: float = 0.3

    # Keep the block flat on the table (do not tip or shove it off).
    off_table_penalty: float = 1.0

    # Keep the block from rotating while being pushed. Penalizes the angular
    # deviation (radians, summed over roll/pitch/yaw) from the block's initial
    # orientation, so a clean straight slide is preferred over a spin.
    rotation_penalty: float = 3.0

    # One-time terminal bonus when the skill's success fact becomes true.
    success_bonus: float = 5.0


def _angular_deviation(rot: np.ndarray | None, reference: np.ndarray | None) -> float:
    """Summed absolute angular difference (radians) from a reference orientation.

    Differences are wrapped to (-pi, pi] so angle wraparound is handled correctly.
    """
    if rot is None or reference is None:
        return 0.0
    delta = np.asarray(rot, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    wrapped = np.arctan2(np.sin(delta), np.cos(delta))
    return float(np.sum(np.abs(wrapped)))


@dataclass
class PushRewardShaper:
    """Stateful shaper for one push direction.

    The direction is fixed at construction from the skill name, so a single
    class implementation serves all four directional push skills.
    """

    skill_name: str
    config: PushRewardConfig = field(default_factory=PushRewardConfig)

    direction: np.ndarray = field(init=False)
    previous_along: float | None = None
    initial_rotation: np.ndarray | None = None
    success_earned: bool = False

    def __post_init__(self) -> None:
        self.direction = require_push_direction(self.skill_name)

    def reset(self) -> None:
        self.previous_along = None
        self.initial_rotation = None
        self.success_earned = False

    def compute(
        self,
        *,
        displacement: np.ndarray,
        xy_dist: float,
        object_rot: np.ndarray | None,
        action: np.ndarray,
        facts: dict[str, bool],
        success: bool,
    ) -> tuple[float, dict[str, float]]:
        cfg = self.config

        action = np.asarray(action, dtype=np.float64)
        disp = np.asarray(displacement, dtype=np.float64)[:2]

        # Capture the resting orientation on the first step so rotation is judged
        # relative to where the block started.
        if self.initial_rotation is None and object_rot is not None:
            self.initial_rotation = np.asarray(object_rot, dtype=np.float64).copy()

        along = float(np.dot(disp, self.direction))
        lateral_vec = disp - along * self.direction
        lateral_dist = float(np.linalg.norm(lateral_vec))
        rotation_dev = _angular_deviation(object_rot, self.initial_rotation)
        on_table = bool(facts.get("object_on_table", False))

        components: dict[str, float] = {
            "time": -cfg.time_penalty,
            "movement": -cfg.action_l2_penalty * float(np.dot(action, action)),
            "progress": 0.0,
            "lateral": -cfg.lateral_penalty * lateral_dist,
            "proximity": -cfg.proximity_weight * float(xy_dist),
            "rotation": -cfg.rotation_penalty * rotation_dev,
            "table": 0.0 if on_table else -cfg.off_table_penalty,
            "success": 0.0,
        }

        # Reward incremental progress along the push direction.
        if self.previous_along is not None:
            progress = np.clip(
                along - self.previous_along,
                -cfg.max_progress,
                cfg.max_progress,
            )
            components["progress"] = cfg.progress_weight * float(progress)

        if success and not self.success_earned:
            components["success"] = cfg.success_bonus
            self.success_earned = True

        self.previous_along = along

        return float(sum(components.values())), components
