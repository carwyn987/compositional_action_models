#!/usr/bin/env python3
"""Entry point: build skills and an environment from command-line options, then train.

    ./main.py --skills pickup putdown --algorithm sac --total-timesteps 200000 --reward shaped
    ./main.py --skills pickup --algorithm ppo --num-blocks 3
    ./main.py --skills pickup putdown --algorithm random --max-episodes 20 --render
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from cam.domain.action_model_library.loader import SYMBOLIC_ACTION_MODEL_FORMATS
from cam.environments.fetch import fetch_multiblock_environment
from cam.experiments.environment_setup import OPERATOR_ENCODERS, setup_environment
from cam.logging_config import configure_logging
from cam.skills.registry import SKILL_REGISTRY, build_skill
from cam.skills.skill import Skill


def parse_args(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    skills = parser.add_argument_group("skills")
    skills.add_argument("--skills", nargs="+", required=True, choices=sorted(SKILL_REGISTRY))
    skills.add_argument(
        "--symbolic-action-model-format",
        default="pddl",
        choices=sorted(SYMBOLIC_ACTION_MODEL_FORMATS),
    )

    environment = parser.add_argument_group("environment")
    environment.add_argument("--environment-id", default=fetch_multiblock_environment.ENVIRONMENT_ID)
    environment.add_argument(
        "--num-blocks",
        type=int,
        help=f"blocks in {fetch_multiblock_environment.ENVIRONMENT_ID} (default 2)",
    )
    environment.add_argument("--render", action="store_true", help="open a viewer window")
    environment.add_argument("--window-width", type=int, default=1280)
    environment.add_argument("--window-height", type=int, default=960)

    training = parser.add_argument_group("training")
    training.add_argument(
        "--algorithm", choices=["sac", "ppo", "random"], default="sac",
        help="sac/ppo: train with Stable-Baselines3; random: run random actions (no learning)",
    )
    training.add_argument("--total-timesteps", type=int, default=100_000, help="environment steps (sac/ppo)")
    training.add_argument("--max-episodes", type=int, default=10, help="episodes to run (random)")
    training.add_argument("--max-steps-per-episode", type=int, default=100)
    training.add_argument("--output-dir", default="outputs", help="runs are saved to <output-dir>/<run name>/")
    training.add_argument("--seed", type=int, default=0)
    training.add_argument(
        "--operator-encoder", choices=sorted(OPERATOR_ENCODERS), default="multi-hot",
        help="how the lifted action model is embedded for the policy (docs/policy_inputs.md)",
    )
    training.add_argument(
        "--max-operator-arity", type=int, default=3,
        help="most parameters any operator may have; fixes the multi-hot vocabulary and grounding slots",
    )
    training.add_argument(
        "--reward", choices=["sparse", "shaped"], default="sparse",
        help="shaped: Fetch shaped reward where one exists (pickup), sparse otherwise",
    )

    args = parser.parse_args(argv)
    if args.num_blocks is not None and args.environment_id != fetch_multiblock_environment.ENVIRONMENT_ID:
        parser.error(f"--num-blocks applies only to --environment-id {fetch_multiblock_environment.ENVIRONMENT_ID}")
    return vars(args)


def run_name(config: dict) -> str:
    """e.g. pickup-putdown_sac_multi-hot_sparse_seed0"""
    return "_".join(
        ["-".join(config["skills"]), config["algorithm"], config["operator_encoder"], config["reward"], f"seed{config['seed']}"]
    )


def train(config: dict, env: gym.Env) -> None:
    """Train a policy with Stable-Baselines3 (imported here so random runs do not need it).

    The run directory also gets config.json, the full training config, so the
    same environment can be rebuilt to evaluate the model (scripts/run_skill.py).
    """
    from cam.training.stable_baselines3_trainer import train_stable_baselines3

    run_directory = Path(config["output_dir"]) / run_name(config)
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "config.json").write_text(json.dumps(config, indent=2))  # lets run_skill.py rebuild the env
    train_stable_baselines3(env, config["algorithm"], config["total_timesteps"], run_directory, seed=config["seed"])


def run_random_actions(config: dict, env: gym.Env, skills: list[Skill]) -> list[float]:
    """Run up to max_episodes episodes of random actions; return episode returns.

    Skills take turns across episodes; the environment sets up each episode and
    ends it once the episode's grounded action succeeds.
    """
    episode_returns = []
    for episode in range(config["max_episodes"]):
        skill = skills[episode % len(skills)]
        obs, info = env.reset(seed=config["seed"] if episode == 0 else None, options={"skill": skill.name})

        episode_return = 0.0
        for step in range(config["max_steps_per_episode"]):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            if terminated or truncated:
                break
        episode_returns.append(episode_return)
        print(
            f"episode {episode:04d} {str(info['grounded_action_model']):20s} steps={step + 1:4d} "
            f"return={episode_return:8.2f} success={bool(info['is_success'])}"
        )
    print(f"mean return over {len(episode_returns)} episodes: {np.mean(episode_returns):.2f}")
    return episode_returns


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    configure_logging()
    skills = [build_skill(skill_name, config) for skill_name in config["skills"]]
    for skill in skills:
        print(skill.symbolic_action_model)

    env = setup_environment(config, skills)
    try:
        if config["algorithm"] == "random":
            run_random_actions(config, env, skills)
        else:
            train(config, env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
