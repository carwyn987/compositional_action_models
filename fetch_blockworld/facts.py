"""Symbolic fact extraction for Gymnasium-Robotics Fetch environments.

The native Fetch object tasks expose a state vector inside obs["observation"].
For object tasks, Gymnasium-Robotics follows the usual Fetch layout:
  0:3   gripper position
  3:6   object position
  6:9   object position relative to gripper
  9:11  gripper finger state
The evaluator also falls back to MuJoCo named bodies/sites where useful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class FetchState:
    grip_pos: Array
    object_pos: Array | None
    achieved_goal: Array | None
    desired_goal: Array | None
    gripper_width: float | None
    initial_object_pos: Array | None
    table_object_z: float | None


class FactEvaluator:
    """Convert low-level Fetch state into boolean symbolic predicates.

    Thresholds are intentionally centralized here so you can tune them without
    touching the RL/training code.
    """

    def __init__(
        self,
        *,
        object_half_size: float = 0.025,
        near_tol: float = 0.055,
        xy_tol: float = 0.045,
        z_tol: float = 0.035,
        goal_tol: float = 0.05,
        push_distance: float = 0.08,
        lift_height: float = 0.06,
        open_width: float = 0.035,
        closed_width: float = 0.020,
    ) -> None:
        self.object_half_size = object_half_size
        self.near_tol = near_tol
        self.xy_tol = xy_tol
        self.z_tol = z_tol
        self.goal_tol = goal_tol
        self.push_distance = push_distance
        self.lift_height = lift_height
        self.open_width = open_width
        self.closed_width = closed_width
        self.initial_object_pos: Array | None = None
        self.table_object_z: float | None = None

    def reset_reference(self, obs: dict[str, Array]) -> None:
        """Remember the reset object pose as the table/resting reference."""
        state = self.state(obs)
        self.initial_object_pos = None if state.object_pos is None else state.object_pos.copy()
        self.table_object_z = None if state.object_pos is None else float(state.object_pos[2])

    def state(self, obs: dict[str, Array]) -> FetchState:
        raw = np.asarray(obs["observation"], dtype=np.float64)
        achieved_goal = np.asarray(obs.get("achieved_goal"), dtype=np.float64) if "achieved_goal" in obs else None
        desired_goal = np.asarray(obs.get("desired_goal"), dtype=np.float64) if "desired_goal" in obs else None

        grip_pos = raw[0:3].copy()
        has_object = raw.shape[0] >= 25
        object_pos = raw[3:6].copy() if has_object else None
        gripper_width = float(np.sum(raw[9:11])) if has_object else None

        return FetchState(
            grip_pos=grip_pos,
            object_pos=object_pos,
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
            gripper_width=gripper_width,
            initial_object_pos=self.initial_object_pos,
            table_object_z=self.table_object_z,
        )

    def facts(self, obs: dict[str, Array]) -> dict[str, bool]:
        s = self.state(obs)
        if s.object_pos is None:
            return {
                "has_object": False,
                "object_on_table": False,
                "object_lifted": False,
                "gripper_near_object": False,
                "gripper_above_object": False,
                "gripper_touching_object_top": False,
                "gripper_open": False,
                "gripper_closed": False,
                "holding_object": False,
                "object_at_goal": False,
                "object_moved_left": False,
                "object_moved_right": False,
                "object_moved_forward": False,
                "object_moved_backward": False,
            }

        obj = s.object_pos
        grip = s.grip_pos
        table_z = s.table_object_z if s.table_object_z is not None else float(obj[2])
        start = s.initial_object_pos if s.initial_object_pos is not None else obj

        grip_obj_dist = float(np.linalg.norm(grip - obj))
        xy_dist = float(np.linalg.norm(grip[:2] - obj[:2]))
        top_z = float(obj[2] + self.object_half_size)
        lift = float(obj[2] - table_z)
        disp = obj - start

        object_at_goal = False
        if s.achieved_goal is not None and s.desired_goal is not None:
            object_at_goal = bool(np.linalg.norm(s.achieved_goal - s.desired_goal) <= self.goal_tol)

        gripper_open = False
        gripper_closed = False
        if s.gripper_width is not None:
            gripper_open = bool(s.gripper_width >= self.open_width)
            gripper_closed = bool(s.gripper_width <= self.closed_width)

        gripper_above = bool(xy_dist <= self.xy_tol and grip[2] >= top_z)
        touching_top = bool(gripper_above and abs(float(grip[2] - top_z)) <= self.z_tol)
        object_lifted = bool(lift >= self.lift_height)
        near_object = bool(grip_obj_dist <= self.near_tol)

        return {
            "has_object": True,
            "object_on_table": bool(obj[2] <= table_z + self.z_tol),
            "object_lifted": object_lifted,
            "gripper_near_object": near_object,
            "gripper_above_object": gripper_above,
            "gripper_touching_object_top": touching_top,
            "gripper_open": gripper_open,
            "gripper_closed": gripper_closed,
            "holding_object": bool(object_lifted and near_object),
            "object_at_goal": object_at_goal,
            # Coordinate convention: x-/x+ and y+/y-. If the viewer looks flipped,
            # rename these predicates, not the underlying policy interface.
            "object_moved_left": bool(disp[0] <= -self.push_distance),
            "object_moved_right": bool(disp[0] >= self.push_distance),
            "object_moved_forward": bool(disp[1] >= self.push_distance),
            "object_moved_backward": bool(disp[1] <= -self.push_distance),
        }

    def is_true(self, obs: dict[str, Array], fact_name: str) -> bool:
        facts = self.facts(obs)
        if fact_name not in facts:
            raise KeyError(f"Unknown fact {fact_name!r}. Known facts: {sorted(facts)}")
        return facts[fact_name]

    def numeric_summary(self, obs: dict[str, Array]) -> dict[str, Any]:
        s = self.state(obs)
        out: dict[str, Any] = {
            "grip_pos": s.grip_pos.round(4).tolist(),
            "object_pos": None if s.object_pos is None else s.object_pos.round(4).tolist(),
            "desired_goal": None if s.desired_goal is None else s.desired_goal.round(4).tolist(),
            "gripper_width": s.gripper_width,
        }
        if s.object_pos is not None and s.initial_object_pos is not None:
            out["object_displacement"] = (s.object_pos - s.initial_object_pos).round(4).tolist()
        return out
