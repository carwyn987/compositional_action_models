"""Train ONE policy on several symbolic Fetch skills at once.

A single Stable-Baselines3 model learns every requested skill (e.g. pickup and
putdown). The skills share the same observation/action space; the *only* signal
that tells the policy which skill to perform is the symbolic embedding vector
appended to the observation. Each episode the environment samples one of the
skills and serves that skill's reward, termination, and embedding.

Because the embedding is the sole differentiator, the embedder must produce a
distinct vector per skill -- this is enforced at startup. (The vanilla, no-
embedding setup therefore cannot do multi-skill training and is intentionally
unsupported here.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor

from fetch_blockworld.envs import make_skill_env
from fetch_blockworld.skills import SKILLS, require_skill
from fetch_blockworld.teacher import (
    TeacherConfig,
    TeacherDataset,
    behavior_clone_policy,
    collect_teacher_dataset,
)
from symb_model_embeddings import (
    EnvWrapper,
    SymbolicActionModel,
    add_embedder_cli_args,
    build_embedder,
    embed_tag,
    embedder_config_from_args,
    model_basename,
)


class MultiSkillEnv(gym.Env):
    """Route each episode to one of several single-skill envs.

    All sub-environments must share an observation and action space (they differ
    only by skill reward/termination and by the embedding their observation
    carries). On every ``reset`` a skill is selected and all subsequent steps go
    to that skill's env until the next reset.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        envs: dict[str, gym.Env],
        *,
        seed: int | None = None,
        strategy: str = "random",
    ) -> None:
        super().__init__()
        if not envs:
            raise ValueError("MultiSkillEnv requires at least one sub-environment")
        self.envs = envs
        self.names = list(envs)
        first = envs[self.names[0]]
        for name, env in envs.items():
            if env.observation_space != first.observation_space:
                raise ValueError(
                    f"Skill {name!r} has a different observation space; multi-skill "
                    "training needs skills that share one observation space."
                )
            if env.action_space != first.action_space:
                raise ValueError(
                    f"Skill {name!r} has a different action space; multi-skill "
                    "training needs skills that share one action space."
                )
        self.observation_space = first.observation_space
        self.action_space = first.action_space
        self.render_mode = getattr(first, "render_mode", None)

        self.strategy = strategy
        self._rng = np.random.default_rng(seed)
        self._cursor = 0
        self._active_name = self.names[0]

    def _pick_skill(self) -> str:
        if self.strategy == "round_robin":
            name = self.names[self._cursor % len(self.names)]
            self._cursor += 1
            return name
        return self.names[int(self._rng.integers(len(self.names)))]

    def reset(self, *, seed=None, options=None):
        self._active_name = self._pick_skill()
        obs, info = self.envs[self._active_name].reset(seed=seed, options=options)
        info = dict(info)
        info["skill"] = self._active_name
        return obs, info

    def step(self, action):
        return self.envs[self._active_name].step(action)

    def render(self):
        return self.envs[self._active_name].render()

    def close(self):
        for env in self.envs.values():
            env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skills",
        nargs="+",
        choices=sorted(SKILLS),
        default=["pickup", "putdown"],
        help="Skills to train into the single shared model.",
    )
    parser.add_argument("--algo", choices=["sac", "ppo"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--sampling",
        choices=["random", "round_robin"],
        default="random",
        help="How each episode chooses which skill to train.",
    )
    parser.add_argument("--logdir", type=Path, default=Path("runs"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--teacher", choices=["bc", "off"], default="bc")
    parser.add_argument("--teacher-rollouts", type=int, default=8)
    parser.add_argument("--teacher-gradient-steps", type=int, default=500)
    parser.add_argument("--teacher-batch-size", type=int, default=256)
    parser.add_argument("--teacher-lr", type=float, default=1e-4)
    add_embedder_cli_args(parser)
    return parser.parse_args()


def build_skill_envs(
    skills: list[str],
    args: argparse.Namespace,
) -> dict[str, gym.Env]:
    """Build one embedding-augmented env per skill, with distinct embeddings."""
    embedder = build_embedder(embedder_config_from_args(args))
    render_mode = "human" if args.render else None

    embeddings: dict[str, np.ndarray] = {}
    envs: dict[str, gym.Env] = {}
    for i, name in enumerate(skills):
        spec = require_skill(name)
        action_model = SymbolicActionModel(name=spec.name, description=spec.action_model)
        embedding = embedder.embed(action_model)
        embeddings[name] = embedding
        envs[name] = EnvWrapper(
            make_skill_env(name, render_mode=render_mode, seed=args.seed + i),
            embedding,
        )

    _assert_distinct_embeddings(embeddings)
    return envs


def _assert_distinct_embeddings(embeddings: dict[str, np.ndarray]) -> None:
    """Fail loudly if any two skills map to the same embedding vector.

    With identical embeddings the shared policy gets no signal distinguishing the
    skills, which defeats the point of multi-skill training.
    """
    items = list(embeddings.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (a_name, a_vec), (b_name, b_vec) = items[i], items[j]
            if np.allclose(a_vec, b_vec):
                raise ValueError(
                    f"Skills {a_name!r} and {b_name!r} produce identical embeddings, "
                    "so the shared policy cannot tell them apart. Use an embedder "
                    "that differs per skill (mock / text-name / text-action_model)."
                )


def warm_start_multiskill(model, envs: dict[str, gym.Env], args: argparse.Namespace) -> None:
    """Behaviour-clone the policy on scripted demos pooled across all skills."""
    observations: list[dict[str, np.ndarray]] = []
    actions: list[np.ndarray] = []
    for i, (name, env) in enumerate(envs.items()):
        spec = require_skill(name)
        config = TeacherConfig(
            rollouts=args.teacher_rollouts,
            max_steps_per_rollout=spec.max_episode_steps,
            gradient_steps=args.teacher_gradient_steps,
            batch_size=args.teacher_batch_size,
            learning_rate=args.teacher_lr,
        )
        dataset = collect_teacher_dataset(
            env, name, config, seed=args.seed + 10_000 + 1_000 * i
        )
        print(f"Collected {dataset.size} teacher examples for skill={name}")
        observations.extend(dataset.observations)
        actions.append(dataset.actions)

    pooled = TeacherDataset(
        observations=observations, actions=np.concatenate(actions, axis=0)
    )
    bc_config = TeacherConfig(
        rollouts=args.teacher_rollouts,
        max_steps_per_rollout=max(require_skill(n).max_episode_steps for n in envs),
        gradient_steps=args.teacher_gradient_steps,
        batch_size=args.teacher_batch_size,
        learning_rate=args.teacher_lr,
    )
    print(f"Behaviour-cloning on {pooled.size} pooled examples across {len(envs)} skills")
    behavior_clone_policy(model, pooled, bc_config)


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.logdir.mkdir(parents=True, exist_ok=True)

    config = embedder_config_from_args(args)
    skill_token = "multi-" + "-".join(args.skills)
    basename = model_basename(
        skill_token, args.algo, embed_tag(config), args.teacher, args.seed
    )

    envs = build_skill_envs(args.skills, args)
    multi_env = MultiSkillEnv(envs, seed=args.seed, strategy=args.sampling)
    env = Monitor(multi_env, filename=str(args.logdir / f"{basename}.monitor.csv"))

    if args.algo == "sac":
        model = SAC(
            "MultiInputPolicy",
            env,
            verbose=1,
            seed=args.seed,
            learning_rate=3e-4,
            buffer_size=300_000,
            learning_starts=1_000,
            batch_size=256,
            gamma=0.95,
            train_freq=1,
            gradient_steps=1,
            tensorboard_log=str(args.logdir),
        )
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            seed=args.seed,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            gamma=0.95,
            tensorboard_log=str(args.logdir),
        )

    if args.teacher == "bc":
        warm_start_multiskill(model, envs, args)
    else:
        print("Teacher warm-start disabled")

    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    out_path = args.models_dir / basename
    model.save(out_path)
    env.close()
    print(f"Saved multi-skill model ({', '.join(args.skills)}) to {out_path}.zip")


if __name__ == "__main__":
    main()
