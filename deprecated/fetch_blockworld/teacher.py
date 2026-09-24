"""Teacher-assisted behavior-cloning warm starts for SB3 policies."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from .envs import FetchSkillEnv
from .scripted import scripted_teacher_action


ObsDict = dict[str, np.ndarray]


@dataclass(frozen=True)
class TeacherConfig:
    rollouts: int = 8
    max_steps_per_rollout: int = 75
    gradient_steps: int = 500
    batch_size: int = 256
    learning_rate: float = 1e-4


@dataclass(frozen=True)
class TeacherDataset:
    observations: list[ObsDict]
    actions: np.ndarray

    @property
    def size(self) -> int:
        return int(self.actions.shape[0])


def collect_teacher_dataset(
    env: gym.Env,
    skill_name: str,
    config: TeacherConfig,
    *,
    seed: int = 0,
) -> TeacherDataset:
    """Roll out the scripted teacher and collect (observation, action) pairs."""

    skill_env = _find_skill_env(env)
    observations: list[ObsDict] = []
    actions: list[np.ndarray] = []

    for ep in range(config.rollouts):
        obs, _info = env.reset(seed=seed + ep)
        for _ in range(config.max_steps_per_rollout):
            action = scripted_teacher_action(skill_name, obs, skill_env.evaluator)
            observations.append(_copy_obs(obs))
            actions.append(np.asarray(action, dtype=np.float32).copy())

            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break

    if not actions:
        raise RuntimeError(f"Teacher produced no examples for skill {skill_name!r}")

    return TeacherDataset(observations=observations, actions=np.stack(actions).astype(np.float32))


def behavior_clone_policy(
    model: object,
    dataset: TeacherDataset,
    config: TeacherConfig,
) -> None:
    """Supervised warm-start: make the SB3 policy imitate teacher actions.

    This updates the policy parameters before RL fine-tuning. It does not alter
    environment transitions or replay-buffer semantics.
    """

    policy = model.policy
    policy.set_training_mode(True)
    device = policy.device
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    n = dataset.size
    batch_size = min(config.batch_size, n)
    rng = np.random.default_rng(0)

    for step in range(config.gradient_steps):
        idx = rng.integers(0, n, size=batch_size)
        batch_obs = _stack_obs(dataset.observations, idx)
        target = torch.as_tensor(dataset.actions[idx], dtype=torch.float32, device=device)

        tensor_obs, _ = policy.obs_to_tensor(batch_obs)
        pred = policy._predict(tensor_obs, deterministic=True)
        if isinstance(pred, tuple):
            pred = pred[0]

        loss = F.mse_loss(pred.float(), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % 100 == 0 or step + 1 == config.gradient_steps:
            print(f"teacher_bc_step={step + 1} loss={loss.item():.6f}")

    policy.set_training_mode(False)


def warm_start_with_teacher(
    model: object,
    env: gym.Env,
    skill_name: str,
    config: TeacherConfig,
    *,
    seed: int = 0,
) -> None:
    dataset = collect_teacher_dataset(env, skill_name, config, seed=seed)
    print(f"Collected {dataset.size} teacher examples for skill={skill_name}")
    behavior_clone_policy(model, dataset, config)


def _find_skill_env(env: gym.Env) -> FetchSkillEnv:
    current = env
    while True:
        if isinstance(current, FetchSkillEnv):
            return current
        next_env = getattr(current, "env", None)
        if next_env is None:
            raise TypeError("Could not find FetchSkillEnv inside wrapped environment")
        current = next_env


def _copy_obs(obs: ObsDict) -> ObsDict:
    return {k: np.asarray(v).copy() for k, v in obs.items()}


def _stack_obs(observations: list[ObsDict], idx: np.ndarray) -> ObsDict:
    keys = observations[0].keys()
    return {k: np.stack([observations[int(i)][k] for i in idx]) for k in keys}
