"""Train a continuous low-level policy for one symbolic Fetch skill.

Same as experimental/train_vanilla_skill.py, but augments the environment
observations with a constant symbolic embedding vector via EnvWrapper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor

from fetch_blockworld.envs import make_skill_env
from fetch_blockworld.skills import SKILLS, require_skill
from fetch_blockworld.teacher import TeacherConfig, warm_start_with_teacher
from symb_model_embeddings import (
    EnvWrapper,
    SymbolicActionModel,
    add_embedder_cli_args,
    basename_from_args,
    build_embedder,
    embedder_config_from_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", choices=sorted(SKILLS), required=True)
    parser.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--logdir", type=Path, default=Path("runs"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--teacher", choices=["bc", "off"], default="bc")
    parser.add_argument("--teacher-rollouts", type=int, default=8)
    parser.add_argument("--teacher-gradient-steps", type=int, default=500)
    parser.add_argument("--teacher-batch-size", type=int, default=256)
    parser.add_argument("--teacher-lr", type=float, default=1e-4)
    add_embedder_cli_args(parser)
    return parser.parse_args()


def build_skill_embedding(skill_name: str, args: argparse.Namespace) -> np.ndarray:
    """Build the symbolic embedding vector injected into the observation."""
    spec = require_skill(skill_name)
    action_model = SymbolicActionModel(name=spec.name, description=spec.action_model)
    embedder = build_embedder(embedder_config_from_args(args))
    return embedder.embed(action_model)


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.logdir.mkdir(parents=True, exist_ok=True)

    basename = basename_from_args(args)

    render_mode = "human" if args.render else None
    skill_env = make_skill_env(args.skill, render_mode=render_mode, seed=args.seed)
    raw_env = EnvWrapper(skill_env, build_skill_embedding(args.skill, args))
    env = Monitor(raw_env, filename=str(args.logdir / f"{basename}.monitor.csv"))

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
        teacher_config = TeacherConfig(
            rollouts=args.teacher_rollouts,
            max_steps_per_rollout=skill_env.skill.max_episode_steps,
            gradient_steps=args.teacher_gradient_steps,
            batch_size=args.teacher_batch_size,
            learning_rate=args.teacher_lr,
        )
        warm_start_with_teacher(model, raw_env, args.skill, teacher_config, seed=args.seed + 10_000)
    else:
        print("Teacher warm-start disabled")

    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    out_path = args.models_dir / basename
    model.save(out_path)
    env.close()
    print(f"Saved model to {out_path}.zip")


if __name__ == "__main__":
    main()
