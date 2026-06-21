"""Environment construction and skill-task wrappers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import gymnasium_robotics
import numpy as np

from .facts import FactEvaluator
from .skills import SkillSpec, require_skill, skill_reward
from .pickup_rewards import PickupRewardShaper


gym.register_envs(gymnasium_robotics)


class FetchSkillEnv(gym.Wrapper):
    """Wrap a native Fetch env as a task for one symbolic skill.

    The underlying action space stays continuous: [dx, dy, dz, gripper].
    The wrapper changes only the reward/termination condition and exposes
    symbolic facts in info["facts"].
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        env: gym.Env,
        skill: SkillSpec,
        evaluator: FactEvaluator | None = None,
    ) -> None:
        super().__init__(env)
        self.skill = skill
        self.evaluator = evaluator or FactEvaluator()

        # Stateful because it tracks progress and one-time milestones.
        self.pickup_reward_shaper = (
            PickupRewardShaper()
            if skill.name == "pickup"
            else None
        )
    
    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        # Native Fetch reset restores the arm and samples a fresh object pose.
        # Do not replace this with object teleporting, otherwise arm/block state
        # can silently carry artifacts across episodes.
        obs, info = self.env.reset(**kwargs)
        self.evaluator.reset_reference(obs)
        
        if self.pickup_reward_shaper is not None:
            self.pickup_reward_shaper.reset()

        scripted_pickup_success = None
        scripted_pickup_steps = 0

        # Make putdown non-trivial: start from an actually grasped/lifted block.
        # Keep the original post-reset table/object reference so object_on_table
        # is judged against the randomized table-resting pose, not the lifted pose.
        if self.skill.name == "putdown":
            from .scripted import run_scripted_pickup

            result = run_scripted_pickup(self.env, obs, self.evaluator)
            obs = result.obs
            scripted_pickup_success = result.success
            scripted_pickup_steps = result.steps

        info = dict(info)
        info["facts"] = self.evaluator.facts(obs)
        info["numeric_state"] = self.evaluator.numeric_summary(obs)
        info["skill"] = self.skill.name
        if scripted_pickup_success is not None:
            info["scripted_pickup_success"] = scripted_pickup_success
            info["scripted_pickup_steps"] = scripted_pickup_steps
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        obs, _native_reward, terminated, truncated, info = self.env.step(action)
        reward, success, facts, reward_components = skill_reward(
            self.skill.name,
            obs,
            self.evaluator,
            action=action,
            pickup_reward_shaper=self.pickup_reward_shaper,
        )

        info = dict(info)
        info["is_success"] = float(success)
        info["facts"] = facts
        info["numeric_state"] = self.evaluator.numeric_summary(obs)
        info["skill"] = self.skill.name
        info["reward_components"] = reward_components

        if self.skill.terminate_on_success and success:
            terminated = True

        return obs, reward, terminated, truncated, info


def make_fetch_env(env_id: str = "FetchPickAndPlace-v4", render_mode: str | None = None, seed: int | None = None) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def make_skill_env(skill_name: str, render_mode: str | None = None, seed: int | None = None) -> FetchSkillEnv:
    spec = require_skill(skill_name)
    base = gym.make(spec.env_id, render_mode=render_mode, max_episode_steps=spec.max_episode_steps)
    env = FetchSkillEnv(base, skill=spec)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def get_current_obs(env: gym.Env, fallback: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fetch envs expose _get_obs(); use fallback if a future version hides it."""
    unwrapped = env.unwrapped
    get_obs = getattr(unwrapped, "_get_obs", None)
    if callable(get_obs):
        return get_obs()
    return fallback
