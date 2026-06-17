"""Small scripted controllers used to prepare skill-specific reset states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from .facts import FactEvaluator


@dataclass(frozen=True)
class ScriptedPickupConfig:
    """Parameters for a simple open-loop/servo pickup initializer."""

    approach_height: float = 0.12
    grasp_height: float = 0.018
    lift_height: float = 0.12
    gain: float = 12.0
    target_tol: float = 0.012
    approach_steps: int = 45
    descend_steps: int = 35
    close_steps: int = 25
    lift_steps: int = 45
    open_cmd: float = 1.0
    close_cmd: float = -1.0


@dataclass(frozen=True)
class ScriptedPickupResult:
    obs: dict[str, np.ndarray]
    success: bool
    facts: dict[str, bool]
    steps: int


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

    obs, n = _servo_to(env, obs, evaluator, approach, cfg.open_cmd, cfg.approach_steps, cfg)
    steps += n
    obs, n = _servo_to(env, obs, evaluator, grasp, cfg.open_cmd, cfg.descend_steps, cfg)
    steps += n
    obs, n = _hold(env, obs, np.array([0.0, 0.0, 0.0, cfg.close_cmd], dtype=np.float32), cfg.close_steps)
    steps += n
    obs, n = _servo_to(env, obs, evaluator, lift, cfg.close_cmd, cfg.lift_steps, cfg)
    steps += n

    facts = evaluator.facts(obs)
    success = bool(facts.get("object_lifted", False) and facts.get("gripper_near_object", False))
    return ScriptedPickupResult(obs=obs, success=success, facts=facts, steps=steps)


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
