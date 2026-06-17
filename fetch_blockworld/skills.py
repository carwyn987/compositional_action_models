"""Skill definitions and shaped rewards for symbolic Fetch actions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .facts import FactEvaluator


@dataclass(frozen=True)
class SkillSpec:
    name: str
    env_id: str
    success_fact: str
    max_episode_steps: int = 75
    terminate_on_success: bool = True


@dataclass(frozen=True)
class RewardConfig:
    """Central reward knobs shared by all symbolic skills.

    Tune action_l2_penalty once to affect every skill. Use
    per_skill_action_l2_penalty for exceptions, e.g.
    {"pushleft": 0.0005, "pickup": 0.002}.
    """

    time_penalty: float = -0.01
    action_l2_penalty: float = 0.001
    per_skill_action_l2_penalty: dict[str, float] = field(default_factory=dict)

    def movement_penalty(self, skill_name: str, action: np.ndarray | None) -> float:
        if action is None:
            return 0.0
        weight = self.per_skill_action_l2_penalty.get(skill_name, self.action_l2_penalty)
        action_arr = np.asarray(action, dtype=np.float32)
        return -float(weight) * float(np.mean(np.square(action_arr)))


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
) -> tuple[float, bool, dict[str, bool]]:
    """Return shaped reward, success flag, and current facts.

    The reward is deliberately simple and local. It is meant to train reusable
    options such as pickup(block) or pushleft(block), not solve the original
    Fetch goal task directly.
    """
    spec = require_skill(skill_name)
    facts = evaluator.facts(obs)
    state = evaluator.state(obs)
    success = bool(facts[spec.success_fact])

    if state.object_pos is None:
        return (-1.0, False, facts)

    obj = state.object_pos
    grip = state.grip_pos
    start = state.initial_object_pos if state.initial_object_pos is not None else obj
    table_z = state.table_object_z if state.table_object_z is not None else float(obj[2])

    grip_to_obj = float(np.linalg.norm(grip - obj))
    xy_dist = float(np.linalg.norm(grip[:2] - obj[:2]))
    lift = float(obj[2] - table_z)
    disp = obj - start

    reward = reward_config.time_penalty
    reward += reward_config.movement_penalty(skill_name, action)

    if skill_name == "pickup":
        target_lift = evaluator.lift_height
        reward += -1.5 * grip_to_obj
        reward += -3.0 * max(0.0, target_lift - lift)
        reward += 0.5 if facts["gripper_above_object"] else 0.0
        reward += 2.0 if success else 0.0

    elif skill_name == "putdown":
        reward += -abs(lift)
        reward += 0.25 if facts["gripper_open"] else 0.0
        reward += 2.0 if success else 0.0

    elif skill_name.startswith("push"):
        direction = {
            "pushleft": np.array([-1.0, 0.0]),
            "pushright": np.array([1.0, 0.0]),
            "pushforward": np.array([0.0, 1.0]),
            "pushbackward": np.array([0.0, -1.0]),
        }[skill_name]
        progress = float(np.dot(disp[:2], direction))
        reward += 5.0 * progress
        reward += -0.3 * xy_dist
        reward += 2.0 if success else 0.0
        reward += -1.0 if not facts["object_on_table"] else 0.0

    elif skill_name == "reach_top":
        top = obj + np.array([0.0, 0.0, evaluator.object_half_size])
        reward += -float(np.linalg.norm(grip - top))
        reward += 2.0 if success else 0.0

    else:
        raise ValueError(f"Unhandled skill: {skill_name}")

    return float(reward), success, facts
