"""Roll out a trained multi-skill policy for one specific skill.

Multi-skill models (trained with train_multiskill.py) use SkillOneHotWrapper +
SkillEmbeddingExtractor instead of EnvWrapper, so rollout_skill.py's observation
space is incompatible with them. This script reconstructs the correct env.

Usage example:
    python rollout_multiskill.py \\
        --model models/multi-pickup-putdown_ppo_mock-d32_t-bc_s0.zip \\
        --skills pickup putdown \\
        --skill pickup \\
        --algo ppo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC

from fetch_blockworld.envs import make_skill_env
from fetch_blockworld.skills import SKILLS, require_skill  # noqa: F401 (side-effect: registers skills)
from symb_model_embeddings.trainable_embedding import SkillEmbeddingExtractor, SkillOneHotWrapper  # noqa: F401 (must be importable for SB3 policy restore)


def _find_embedding_extractor(model) -> SkillEmbeddingExtractor:
    for module in model.policy.modules():
        if isinstance(module, SkillEmbeddingExtractor):
            return module
    raise RuntimeError("SkillEmbeddingExtractor not found in the policy")


def print_embedding_diagnostics(model, skills: list[str], active_skill: str) -> None:
    extractor = _find_embedding_extractor(model)
    table = extractor.current_embeddings()  # (num_skills, dim)
    print(f"Embedding table: {table.shape[0]} skills x {table.shape[1]} dims")
    for i, name in enumerate(skills):
        norm = float(np.linalg.norm(table[i]))
        marker = " <-- active" if name == active_skill else ""
        print(f"  [{i}] {name:20s}  norm={norm:.4f}  vec[:4]={table[i, :4].tolist()}{marker}")
    for i in range(len(skills)):
        for j in range(i + 1, len(skills)):
            a, b = table[i], table[j]
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            print(f"  cos_sim({skills[i]}, {skills[j]}) = {cos:.4f}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--skills",
        nargs="+",
        choices=sorted(SKILLS),
        default=["pickup", "putdown"],
        help="Ordered list of skills the model was trained on (order determines skill index).",
    )
    parser.add_argument("--skill", choices=sorted(SKILLS), required=True,
                        help="Which skill to evaluate.")
    parser.add_argument("--algo", choices=["sac", "ppo"], default="ppo")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.skill not in args.skills:
        raise ValueError(
            f"--skill {args.skill!r} must be in --skills {args.skills}. "
            "The skills list must match what the model was trained on."
        )

    skill_index = args.skills.index(args.skill)
    num_skills = len(args.skills)
    render_mode = None if args.no_render else "human"

    env = SkillOneHotWrapper(
        make_skill_env(args.skill, render_mode=render_mode, seed=args.seed),
        skill_index=skill_index,
        num_skills=num_skills,
    )

    model_cls = SAC if args.algo == "sac" else PPO
    model = model_cls.load(args.model, env=env)
    print_embedding_diagnostics(model, args.skills, args.skill)

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
