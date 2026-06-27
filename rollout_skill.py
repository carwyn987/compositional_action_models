"""Render or inspect a trained symbolic skill policy.

Same as experimental/rollout_vanilla_skill.py, but augments the environment
observations with a constant symbolic embedding vector via EnvWrapper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO, SAC

from fetch_blockworld.envs import make_skill_env
from fetch_blockworld.skills import SKILLS, require_skill
from symb_model_embeddings import (
    EnvWrapper,
    SymbolicActionModel,
    add_embedder_cli_args,
    build_embedder,
    embedder_config_from_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", choices=sorted(SKILLS), required=True)
    parser.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    add_embedder_cli_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = require_skill(args.skill)
    action_model = SymbolicActionModel(name=spec.name, description=spec.action_model)
    embedder = build_embedder(embedder_config_from_args(args))
    env = EnvWrapper(
        make_skill_env(args.skill, render_mode=None if args.no_render else "human", seed=args.seed),
        embedder.embed(action_model),
    )
    model_cls = SAC if args.algo == "sac" else PPO
    model = model_cls.load(args.model, env=env)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
            step += 1
        print(f"episode={ep} steps={step} return={total_reward:.3f} success={info.get('is_success')}")
        print("facts:", info.get("facts"))
        print("numeric_state:", info.get("numeric_state"))

    env.close()


if __name__ == "__main__":
    main()
