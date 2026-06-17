"""Small scripted controllers used for reset preparation and teacher actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from .facts import FactEvaluator


@dataclass(frozen=True)
class ScriptedPickupConfig:
    """Parameters for a simple servo pickup initializer.

    Important: Fetch object_pos is the block center. For a side grasp, the
    gripper site needs to descend to roughly the block center, not merely the
    top face. A small negative grasp_height is often more reliable than a
    positive one because the MuJoCo gripper site is between the fingers.
    """

    approach_height: float = 0.14
    grasp_height: float = -0.005
    lift_height: float = 0.16
    gain: float = 12.0
    target_tol: float = 0.010
    approach_steps: int = 60
    descend_steps: int = 70
    close_steps: int = 40
    lift_steps: int = 80
    open_cmd: float = 1.0
    close_cmd: float = -1.0


@dataclass(frozen=True)
class ScriptedTeacherConfig:
    """Shared knobs for one-step scripted skill teachers."""

    gain: float = 8.0
    open_cmd: float = 1.0
    close_cmd: float = -1.0
    pickup_approach_height: float = 0.14
    pickup_grasp_height: float = 0.0
    pickup_lift_height: float = 0.20
    pickup_xy_tol: float = 0.020
    pickup_z_tol: float = 0.008
    gain: float = 6.0
    putdown_release_height: float = 0.045
    push_behind_dist: float = 0.075
    push_speed: float = 1.0
    push_align_tol: float = 0.035
    push_contact_z_offset: float = 0.0


@dataclass(frozen=True)
class ScriptedPickupResult:
    obs: dict[str, np.ndarray]
    success: bool
    facts: dict[str, bool]
    steps: int


def scripted_teacher_action(
    skill_name: str,
    obs: dict[str, np.ndarray],
    evaluator: FactEvaluator,
    config: ScriptedTeacherConfig | None = None,
) -> np.ndarray:
    """Return one low-level action from a simple teacher for a symbolic skill.

    This is intentionally stateless so it can be used for behavior-cloning
    warm starts. It maps the current Fetch observation to the same continuous
    action space used by the RL policy: [dx, dy, dz, gripper].
    """

    cfg = config or ScriptedTeacherConfig()
    facts = evaluator.facts(obs)
    state = evaluator.state(obs)
    if state.object_pos is None:
        return np.zeros(4, dtype=np.float32)

    obj = state.object_pos
    grip = state.grip_pos
    table_z = state.table_object_z if state.table_object_z is not None else float(obj[2])

    if skill_name == "pickup":
        return _teacher_pickup(grip, obj, facts, cfg)

    if skill_name == "putdown":
        return _teacher_putdown(grip, obj, table_z, facts, evaluator, cfg)

    if skill_name.startswith("push"):
        return _teacher_push(skill_name, grip, obj, cfg)

    if skill_name == "reach_top":
        target = obj + np.array([0.0, 0.0, evaluator.object_half_size], dtype=np.float64)
        return _servo_action(grip, target, gripper_cmd=0.0, gain=cfg.gain)

    raise ValueError(f"No scripted teacher for skill {skill_name!r}")


def run_scripted_pickup(
    env: gym.Env,
    obs: dict[str, np.ndarray],
    evaluator: FactEvaluator,
    config: ScriptedPickupConfig | None = None,
) -> ScriptedPickupResult:
    """Try to put the block in the Fetch gripper using low-level actions.

    This is intended for reset-state preparation, e.g. starting putdown(block)
    episodes from a genuine held-block state. It deliberately steps the
    unwrapped MuJoCo env so Gymnasium's TimeLimit wrapper is not consumed before
    the RL episode starts.
    """

    cfg = config or ScriptedPickupConfig()
    steps = 0
    state = evaluator.state(obs)
    if state.object_pos is None:
        return ScriptedPickupResult(obs=obs, success=False, facts=evaluator.facts(obs), steps=0)

    obj = state.object_pos.copy()
    approach = obj + np.array([0.0, 0.0, cfg.approach_height], dtype=np.float64)
    grasp = obj + np.array([0.0, 0.0, cfg.grasp_height], dtype=np.float64)
    lift = obj + np.array([0.0, 0.0, cfg.lift_height], dtype=np.float64)

    # Open while moving above the randomized block.
    obs, n = _servo_to(env, obs, evaluator, approach, cfg.open_cmd, cfg.approach_steps, cfg)
    steps += n

    # Descend to the block center/slightly below top, not just the top face.
    obs, n = _servo_to(env, obs, evaluator, grasp, cfg.open_cmd, cfg.descend_steps, cfg)
    steps += n

    # Close in place long enough for the finger joints to make contact.
    obs, n = _hold(env, obs, np.array([0.0, 0.0, 0.0, cfg.close_cmd], dtype=np.float32), cfg.close_steps)
    steps += n

    # Lift while continuing to close.
    obs, n = _servo_to(env, obs, evaluator, lift, cfg.close_cmd, cfg.lift_steps, cfg)
    steps += n

    # A short settling hold improves the final observation/facts.
    obs, n = _hold(env, obs, np.array([0.0, 0.0, 0.0, cfg.close_cmd], dtype=np.float32), 10)
    steps += n

    facts = evaluator.facts(obs)
    success = bool(facts.get("object_lifted", False) and facts.get("gripper_near_object", False))
    return ScriptedPickupResult(obs=obs, success=success, facts=facts, steps=steps)


def _teacher_pickup(
    grip: np.ndarray,
    obj: np.ndarray,
    facts: dict[str, bool],
    cfg: ScriptedTeacherConfig,
) -> np.ndarray:
    xy_dist = float(np.linalg.norm(grip[:2] - obj[:2]))

    approach = obj + np.array([0.0, 0.0, cfg.pickup_approach_height], dtype=np.float64)
    grasp = obj + np.array([0.0, 0.0, cfg.pickup_grasp_height], dtype=np.float64)
    lift = obj + np.array([0.0, 0.0, cfg.pickup_lift_height], dtype=np.float64)

    # 1. Move above the block with gripper open.
    if xy_dist > cfg.pickup_xy_tol:
        return _servo_action(grip, approach, cfg.open_cmd, cfg.gain)

    # 2. Descend with gripper open until at grasp height.
    if grip[2] > grasp[2] + cfg.pickup_z_tol:
        return _servo_action(grip, grasp, cfg.open_cmd, cfg.gain)

    # 3. Dedicated grip phase.
    # Stay at grasp height and close. Do not lift yet.
    if not facts.get("gripper_closed", False):
        return _servo_action(grip, grasp, cfg.close_cmd, cfg.gain)

    # 4. Lift phase.
    # Only lift once the gripper is closed.
    return _servo_action(grip, lift, cfg.close_cmd, cfg.gain)


def _teacher_putdown(
    grip: np.ndarray,
    obj: np.ndarray,
    table_z: float,
    facts: dict[str, bool],
    evaluator: FactEvaluator,
    cfg: ScriptedTeacherConfig,
) -> np.ndarray:
    if not facts["object_on_table"]:
        target = np.array(
            [obj[0], obj[1], table_z + evaluator.object_half_size + cfg.putdown_release_height],
            dtype=np.float64,
        )
        return _servo_action(grip, target, cfg.close_cmd, cfg.gain)

    return np.array([0.0, 0.0, 0.0, cfg.open_cmd], dtype=np.float32)


def _teacher_push(
    skill_name: str,
    grip: np.ndarray,
    obj: np.ndarray,
    cfg: ScriptedTeacherConfig,
) -> np.ndarray:
    direction = {
        "pushleft": np.array([-1.0, 0.0], dtype=np.float64),
        "pushright": np.array([1.0, 0.0], dtype=np.float64),
        "pushforward": np.array([0.0, 1.0], dtype=np.float64),
        "pushbackward": np.array([0.0, -1.0], dtype=np.float64),
    }[skill_name]

    behind_xy = obj[:2] - cfg.push_behind_dist * direction
    contact_z = obj[2] + cfg.push_contact_z_offset
    target = np.array([behind_xy[0], behind_xy[1], contact_z], dtype=np.float64)

    xy_err = behind_xy - grip[:2]
    z_err = float(contact_z - grip[2])
    if float(np.linalg.norm(xy_err)) > cfg.push_align_tol or abs(z_err) > 0.025:
        return _servo_action(grip, target, gripper_cmd=0.0, gain=cfg.gain)

    action = np.zeros(4, dtype=np.float32)
    action[:2] = (cfg.push_speed * direction).astype(np.float32)
    return action


def _servo_action(grip: np.ndarray, target: np.ndarray, gripper_cmd: float, gain: float) -> np.ndarray:
    action = np.zeros(4, dtype=np.float32)
    delta = np.asarray(target, dtype=np.float64) - np.asarray(grip, dtype=np.float64)
    action[:3] = np.clip(gain * delta, -1.0, 1.0).astype(np.float32)
    action[3] = np.float32(gripper_cmd)
    return action


def _servo_to(
    env: gym.Env,
    obs: dict[str, np.ndarray],
    evaluator: FactEvaluator,
    target: np.ndarray,
    gripper_cmd: float,
    max_steps: int,
    cfg: ScriptedPickupConfig,
) -> tuple[dict[str, np.ndarray], int]:
    steps = 0
    for _ in range(max_steps):
        grip = evaluator.state(obs).grip_pos
        delta = np.asarray(target, dtype=np.float64) - grip
        action = np.zeros(4, dtype=np.float32)
        action[:3] = np.clip(cfg.gain * delta, -1.0, 1.0).astype(np.float32)
        action[3] = np.float32(gripper_cmd)
        obs = _step_unwrapped(env, action, obs)
        steps += 1
        if float(np.linalg.norm(delta)) <= cfg.target_tol:
            break
    return obs, steps


def _hold(
    env: gym.Env,
    obs: dict[str, np.ndarray],
    action: np.ndarray,
    steps: int,
) -> tuple[dict[str, np.ndarray], int]:
    for _ in range(steps):
        obs = _step_unwrapped(env, action, obs)
    return obs, steps


def _step_unwrapped(env: gym.Env, action: np.ndarray, fallback: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Step the base env without advancing Gymnasium TimeLimit bookkeeping."""
    try:
        obs, _reward, _terminated, _truncated, _info = env.unwrapped.step(action)
    except Exception:
        return fallback

    if isinstance(obs, dict):
        return obs
    return _get_current_obs(env, fallback)


def _get_current_obs(env: gym.Env, fallback: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    get_obs: Any = getattr(env.unwrapped, "_get_obs", None)
    if callable(get_obs):
        obs = get_obs()
        if isinstance(obs, dict):
            return obs
    return fallback
