"""Environment construction and skill-task wrappers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import gymnasium_robotics
import numpy as np

from .facts import FactEvaluator
from .scripted import ScriptedPickupConfig, run_scripted_pickup
from .skills import SkillSpec, require_skill, skill_reward


gym.register_envs(gymnasium_robotics)


class FetchSkillEnv(gym.Wrapper):
    """Wrap a native Fetch env as a task for one symbolic skill.

    The underlying action space stays continuous: [dx, dy, dz, gripper].
    The wrapper changes only the reward/termination condition and exposes
    symbolic facts in info["facts"].
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, env: gym.Env, skill: SkillSpec, evaluator: FactEvaluator | None = None) -> None:
        super().__init__(env)
        self.skill = skill
        self.evaluator = evaluator or FactEvaluator()
        self.scripted_pickup_config = ScriptedPickupConfig()

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.evaluator.reset_reference(obs)

        scripted_info: dict[str, Any] = {}
        if self.skill.name == "putdown":
            table_z = self.evaluator.table_object_z
            result = run_scripted_pickup(self.env, obs, self.evaluator, self.scripted_pickup_config)
            obs = result.obs
            # Keep the table/resting reference from before pickup, otherwise a
            # held block would incorrectly become the new "on table" height.
            self.evaluator.reset_reference(obs)
            self.evaluator.table_object_z = table_z
            scripted_info = {
                "scripted_pickup_success": result.success,
                "scripted_pickup_steps": result.steps,
            }

        info = dict(info)
        info.update(scripted_info)
        info["facts"] = self.evaluator.facts(obs)
        info["numeric_state"] = self.evaluator.numeric_summary(obs)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        obs, _native_reward, terminated, truncated, info = self.env.step(action)
        reward, success, facts = skill_reward(self.skill.name, obs, self.evaluator, action=action)

        info = dict(info)
        info["is_success"] = float(success)
        info["facts"] = facts
        info["numeric_state"] = self.evaluator.numeric_summary(obs)
        info["skill"] = self.skill.name

        if self.skill.terminate_on_success and success:
            terminated = True

        return obs, reward, terminated, truncated, info


def make_fetch_env(env_id: str = "FetchPickAndPlace-v4", render_mode: str | None = None, seed: int | None = None) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_skill_env(skill_name: str, render_mode: str | None = None, seed: int | None = None) -> FetchSkillEnv:
    spec = require_skill(skill_name)
    base = gym.make(spec.env_id, render_mode=render_mode, max_episode_steps=spec.max_episode_steps)
    env = FetchSkillEnv(base, skill=spec)
    if seed is not None:
        env.reset(seed=seed)
    return env


def get_current_obs(env: gym.Env, fallback: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fetch envs expose _get_obs(); use fallback if a future version hides it."""
    unwrapped = env.unwrapped
    get_obs = getattr(unwrapped, "_get_obs", None)
    if callable(get_obs):
        return get_obs()
    return fallback


def set_object_position(env: gym.Env, pos: np.ndarray, joint_name: str = "object0:joint") -> bool:
    """Set the Fetch object free-joint position if MuJoCo internals are available."""
    try:
        import mujoco
    except Exception:
        return False

    try:
        unwrapped = env.unwrapped
        model = unwrapped.model
        data = unwrapped.data
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            return False

        qpos_addr = int(model.jnt_qposadr[joint_id])
        qvel_addr = int(model.jnt_dofadr[joint_id])
        joint_type = int(model.jnt_type[joint_id])

        # Free joint: [x, y, z, qw, qx, qy, qz] and 6 velocity dofs.
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            data.qpos[qpos_addr : qpos_addr + 3] = np.asarray(pos, dtype=np.float64)
            data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
            data.qvel[qvel_addr : qvel_addr + 6] = 0.0
            mujoco.mj_forward(model, data)
            return True
    except Exception:
        return False

    return False
