#!/usr/bin/env python3
"""Entry point: build skills and an environment from command-line options, then train.

    ./main.py --skills pickup
    ./main.py --skills pickup putdown --num-blocks 3 --max-episodes 20 --render
    ./main.py --skills pickup --environment-id FetchPickAndPlace-v4
"""

import argparse

import gymnasium as gym
import numpy as np

from cam.domain.action_model_library.loader import SYMBOLIC_ACTION_MODEL_FORMATS
from cam.environments.fetch import fetch_multiblock_environment
from cam.environments.fetch.fetch_env_state_annotation_wrapper import FetchEnvStateAnnotationWrapper
from cam.environments.fetch.fetch_predicate_evaluation_wrapper import FetchPredicateEvaluationWrapper
from cam.environments.fetch.fetch_scripted_policies import FetchScriptedPickupPolicy, FetchScriptedPutdownPolicy
from cam.environments.skill_environment import SkillEnvironment
from cam.logging_config import configure_logging
from cam.skills.registry import SKILL_REGISTRY, build_skill
from cam.skills.skill import Skill

SETUP_AND_EPISODE_STEP_LIMIT = 10_000  # inner limit; the episode limit is applied outside SkillEnvironment


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
    training.add_argument("--max-episodes", type=int, default=10)
    training.add_argument("--max-steps-per-episode", type=int, default=100)
    training.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    if args.num_blocks is not None and args.environment_id != fetch_multiblock_environment.ENVIRONMENT_ID:
        parser.error(f"--num-blocks applies only to --environment-id {fetch_multiblock_environment.ENVIRONMENT_ID}")
    return vars(args)


def setup_environment(config: dict, skills: list[Skill]) -> gym.Env:
    """Fetch environment annotated with state and facts, running episodes of the given skills.

    Wrapping, inside out: gym.make → FetchEnvStateAnnotationWrapper (environment_state, objects)
    → FetchPredicateEvaluationWrapper (facts) → SkillEnvironment (setup chain, grounding, success,
    reward) → TimeLimit (max_steps_per_episode, setup steps excluded).
    """
    environment_kwargs = {}
    if config["num_blocks"] is not None:
        environment_kwargs["num_blocks"] = config["num_blocks"]
    if config["render"]:
        environment_kwargs.update(width=config["window_width"], height=config["window_height"])

    base_env = gym.make(
        config["environment_id"],
        render_mode="human" if config["render"] else None,
        max_episode_steps=SETUP_AND_EPISODE_STEP_LIMIT,
        **environment_kwargs,
    )
    setup_policies = {"pickup": FetchScriptedPickupPolicy(), "putdown": FetchScriptedPutdownPolicy()}
    skill_env = SkillEnvironment(
        FetchPredicateEvaluationWrapper(FetchEnvStateAnnotationWrapper(base_env)), skills, setup_policies
    )
    env = gym.wrappers.TimeLimit(skill_env, config["max_steps_per_episode"])
    env.action_space.seed(config["seed"])
    return env


def train(config: dict, env: gym.Env, skills: list[Skill]) -> list[float]:
    """Run up to max_episodes episodes of up to max_steps_per_episode steps; return episode returns.

    Skills take turns across episodes; the environment sets up each episode and
    ends it once the episode's grounded action succeeds.
    """
    episode_returns = []
    for episode in range(config["max_episodes"]):
        skill = skills[episode % len(skills)]
        obs, info = env.reset(seed=config["seed"] if episode == 0 else None, options={"skill": skill.name})

        episode_return = 0.0
        for step in range(config["max_steps_per_episode"]):
            action = env.action_space.sample()  # TODO: replace with the policy's action for obs
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
        train(config, env, skills)
    finally:
        env.close()


if __name__ == "__main__":
    main()
