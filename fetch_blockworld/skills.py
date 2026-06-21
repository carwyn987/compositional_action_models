"""Skill definitions and shaped rewards for symbolic Fetch actions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .facts import FactEvaluator
from .pickup_rewards import PickupRewardShaper

@dataclass(frozen=True)
class SkillSpec:
    name: str
    env_id: str
    success_fact: str
    max_episode_steps: int = 10000
    terminate_on_success: bool = True


@dataclass(frozen=True)
class RewardConfig:
    time_penalty: float = -0.01
    action_l2_penalty: float = 0.001
    per_skill_action_l2_penalty: dict[str, float] = field(default_factory=dict)

    def action_penalty_for(self, skill_name: str) -> float:
        return self.per_skill_action_l2_penalty.get(skill_name, self.action_l2_penalty)


DEFAULT_REWARD_CONFIG = RewardConfig()


SKILLS: dict[str, SkillSpec] = {
    "pickup": SkillSpec("pickup", "FetchPickAndPlace-v4", "object_lifted"),
    "putdown": SkillSpec("putdown", "FetchPickAndPlace-v4", "object_on_table"),
    "pushleft": SkillSpec("pushleft", "FetchPush-v4", "object_moved_left"),
    "pushright": SkillSpec("pushright", "FetchPush-v4", "object_moved_right"),
    "pushforward": SkillSpec("pushforward", "FetchPush-v4", "object_moved_forward"),
    "pushbackward": SkillSpec("pushbackward", "FetchPush-v4", "object_moved_backward"),
    "reach_top": SkillSpec("reach_top", "FetchPickAndPlace-v4", "gripper_touching_object_top"),
}


def require_skill(skill_name: str) -> SkillSpec:
    try:
        return SKILLS[skill_name]
    except KeyError as exc:
        raise KeyError(f"Unknown skill {skill_name!r}. Choices: {sorted(SKILLS)}") from exc


def skill_reward(
    skill_name: str,
    obs: dict[str, np.ndarray],
    evaluator: FactEvaluator,
    *,
    action: np.ndarray | None = None,
    reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
    pickup_reward_shaper: PickupRewardShaper | None = None,
) -> tuple[float, bool, dict[str, bool], dict[str, float]]:
    """Return reward, success flag, facts, and reward components."""

    spec = require_skill(skill_name)
    facts = evaluator.facts(obs)
    state = evaluator.state(obs)
    success = bool(facts[spec.success_fact])

    if state.object_pos is None:
        return -1.0, False, facts, {"invalid_state": -1.0, "total": -1.0}

    obj = state.object_pos
    grip = state.grip_pos
    start = (
        state.initial_object_pos
        if state.initial_object_pos is not None
        else obj
    )
    table_z = (
        state.table_object_z
        if state.table_object_z is not None
        else float(obj[2])
    )

    grip_to_obj = float(np.linalg.norm(grip - obj))
    xy_dist = float(np.linalg.norm(grip[:2] - obj[:2]))
    lift = max(0.0, float(obj[2] - table_z))
    disp = obj - start

    action_array = (
        np.zeros(4, dtype=np.float32)
        if action is None
        else np.asarray(action, dtype=np.float32)
    )

    # Pickup has its own stateful, staged reward.
    # Do not apply the generic time/action costs here because the pickup
    # shaper already includes them.
    if skill_name == "pickup":
        if pickup_reward_shaper is None:
            raise RuntimeError(
                "pickup_reward_shaper must be provided for the pickup skill"
            )

        reward, components = pickup_reward_shaper.compute(
            grip_pos=grip,
            object_pos=obj,
            gripper_width=float(state.gripper_width),
            lift=lift,
            target_lift=float(evaluator.lift_height),
            action=action_array,
            facts=facts,
            success=success,
        )

        components["total"] = float(reward)
        return float(reward), success, facts, components

    # Generic costs for all non-pickup skills.
    time_reward = float(reward_config.time_penalty)
    movement_reward = -(
        reward_config.action_penalty_for(skill_name)
        * float(np.dot(action_array, action_array))
    )

    reward = time_reward + movement_reward

    components: dict[str, float] = {
        "time": time_reward,
        "movement": movement_reward,
    }

    if skill_name == "putdown":
        table_reward = -abs(lift)
        open_reward = 0.25 if facts["gripper_open"] else 0.0
        success_reward = 2.0 if success else 0.0

        reward += table_reward + open_reward + success_reward

        components.update(
            {
                "table": table_reward,
                "open": open_reward,
                "success": success_reward,
            }
        )

    elif skill_name.startswith("push"):
        direction = {
            "pushleft": np.array([-1.0, 0.0]),
            "pushright": np.array([1.0, 0.0]),
            "pushforward": np.array([0.0, 1.0]),
            "pushbackward": np.array([0.0, -1.0]),
        }[skill_name]

        progress = float(np.dot(disp[:2], direction))

        progress_reward = 5.0 * progress
        proximity_reward = -0.3 * xy_dist
        success_reward = 2.0 if success else 0.0
        table_reward = -1.0 if not facts["object_on_table"] else 0.0

        reward += (
            progress_reward
            + proximity_reward
            + success_reward
            + table_reward
        )

        components.update(
            {
                "progress": progress_reward,
                "proximity": proximity_reward,
                "table": table_reward,
                "success": success_reward,
            }
        )

    elif skill_name == "reach_top":
        top = obj + np.array(
            [0.0, 0.0, evaluator.object_half_size],
            dtype=np.float64,
        )

        distance_reward = -float(np.linalg.norm(grip - top))
        success_reward = 2.0 if success else 0.0

        reward += distance_reward + success_reward

        components.update(
            {
                "distance": distance_reward,
                "success": success_reward,
            }
        )

    else:
        raise ValueError(f"Unhandled skill: {skill_name}")

    components["total"] = float(reward)
    return float(reward), success, facts, components