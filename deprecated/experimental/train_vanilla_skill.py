"""Train a continuous low-level policy for one symbolic Fetch skill."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fetch_blockworld.envs import make_skill_env
from fetch_blockworld.skills import SKILLS
from fetch_blockworld.teacher import TeacherConfig, warm_start_with_teacher
from symb_model_embeddings import model_basename


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.logdir.mkdir(parents=True, exist_ok=True)

    basename = model_basename(args.skill, args.algo, "vanilla", args.teacher, args.seed)

    render_mode = "human" if args.render else None
    raw_env = make_skill_env(args.skill, render_mode=render_mode, seed=args.seed)
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
            max_steps_per_rollout=raw_env.skill.max_episode_steps,
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
