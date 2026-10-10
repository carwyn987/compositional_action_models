#!/usr/bin/env python3
"""Run episodes of one skill and report setup, facts and success, with a scripted, random or trained policy.

    ./scripts/run_skill.py                                 # putdown, set up by pickup, scripted policy
    ./scripts/run_skill.py --skill pickup --num-blocks 5 --render
    ./scripts/run_skill.py --skill unstack --render        # set up by pickup, then stack
    ./scripts/run_skill.py --policy random --episodes 10
    ./scripts/run_skill.py --policy model --run-directory outputs/pickup_ppo_multi-hot_shaped_seed0 --render

The environment is built by cam.experiments.environment_setup, the same as for
training. With --policy model, its options (blocks, operator encoder, maximum
operator arity, ...) come from the run's config.json, so the observation
matches what the model was trained on; --skill defaults to the first trained skill.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from cam.domain.action_model_library.loader import SYMBOLIC_ACTION_MODEL_FORMATS
from cam.environments.fetch import fetch_multiblock_environment
from cam.environments.fetch.fetch_scripted_policies import (
    FetchScriptedPickupPolicy,
    FetchScriptedPutdownPolicy,
    FetchScriptedStackPolicy,
)
from cam.experiments.environment_setup import setup_environment
from cam.logging_config import configure_logging
from cam.policies.policy import RandomPolicy
from cam.skills.registry import SKILL_REGISTRY, build_skill

logger = logging.getLogger("cam.scripts.run_skill")

SCRIPTED_POLICIES = {
    "pickup": FetchScriptedPickupPolicy,
    "putdown": FetchScriptedPutdownPolicy,
    "stack": FetchScriptedStackPolicy,
    "unstack": FetchScriptedPickupPolicy,  # grasp and lift, from on top of the other block
}
DEFAULT_CONFIG = {  # main.py's defaults, for runs without a trained model
    "symbolic_action_model_format": "pddl",
    "environment_id": fetch_multiblock_environment.ENVIRONMENT_ID,
    "num_blocks": 3,
    "reward": "sparse",
    "operator_encoder": "multi-hot",
    "operator_embedding_dim": 128,
    "text_backend": "mock",
    "operator_text": "pddl",
    "text_embedding_cache": "outputs/text_embedding_cache.json",
    "max_operator_arity": 3,
    "max_steps_per_episode": 100,
    "seed": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skill", choices=sorted(SKILL_REGISTRY), help="default: putdown (model: first trained skill)")
    parser.add_argument("--policy", choices=["scripted", "random", "model"], default="scripted")
    parser.add_argument("--run-directory", type=Path, help="training run with model.zip (required for --policy model)")
    parser.add_argument(
        "--stochastic", action="store_true", help="model: sample actions instead of taking the most likely one"
    )
    parser.add_argument("--symbolic-action-model-format", choices=sorted(SYMBOLIC_ACTION_MODEL_FORMATS))
    parser.add_argument("--num-blocks", type=int, help="default 3 (model: the trained number; must match)")
    parser.add_argument(
        "--reward", choices=["sparse", "shaped"],
        help="shaped: Fetch shaped reward where one exists (pickup, stack, unstack), sparse otherwise; default sparse (model: trained)",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, help="episode step limit, setup steps excluded (default 100 / trained)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.02, help="seconds between rendered frames")
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=960)
    args = parser.parse_args()
    if args.policy == "model" and args.run_directory is None:
        parser.error("--policy model needs --run-directory")
    return args


def load_run_config(run_directory: Path) -> dict:
    """The training config saved with the run; for runs saved before config.json, the options
    encoded in the run name (<skills>_<algorithm>_<encoder>_<reward>_seed<n>) over main.py's defaults."""
    config_path = run_directory / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    skills, algorithm, operator_encoder, reward, seed = run_directory.name.split("_")
    logger.warning("%s has no config.json; using the run name and default options", run_directory)
    return DEFAULT_CONFIG | {
        "skills": skills.split("-"),
        "algorithm": algorithm,
        "operator_encoder": operator_encoder,
        "reward": reward,
        "seed": int(seed.removeprefix("seed")),
        "num_blocks": None,
    }


def build_config(args: argparse.Namespace) -> tuple[dict, str]:
    """(environment config, skill to run): the trained run's config (model) or the defaults, with this
    script's options applied. For a model the environment keeps the trained skills, in training order, so
    the operator encoder is built as in training; a skill outside them is added after them."""
    if args.policy == "model":
        config = load_run_config(args.run_directory)
        if args.num_blocks is not None and args.num_blocks != config["num_blocks"]:
            raise SystemExit(f"--num-blocks {args.num_blocks} differs from the trained {config['num_blocks']}")
        skill = args.skill or config["skills"][0]
    else:
        config = dict(DEFAULT_CONFIG)
        if args.num_blocks is not None:
            config["num_blocks"] = args.num_blocks
        skill = args.skill or "putdown"
    overrides = {
        "symbolic_action_model_format": args.symbolic_action_model_format,
        "reward": args.reward,
        "max_steps_per_episode": args.max_steps,
    }
    environment_skills = list(config["skills"]) if args.policy == "model" else []
    if skill not in environment_skills:
        environment_skills.append(skill)
    return config | {key: value for key, value in overrides.items() if value is not None} | {
        "skills": environment_skills,
        "seed": args.seed,
        "render": args.render,
        "window_width": args.window_width,
        "window_height": args.window_height,
    }, skill


def build_policy(args: argparse.Namespace, config: dict, skill_name: str, env):
    if args.policy == "scripted":
        return SCRIPTED_POLICIES[skill_name]()
    if args.policy == "random":
        return RandomPolicy(env.action_space)
    from cam.policies.stable_baselines3_policy import StableBaselines3Policy  # needs Stable-Baselines3

    if not (args.run_directory / "model.zip").exists():
        raise SystemExit(f"{args.run_directory} has no model.zip (training saves it when it finishes)")
    return StableBaselines3Policy.load(
        args.run_directory / "model.zip", config["algorithm"], deterministic=not args.stochastic
    )


def main() -> None:
    args = parse_args()
    configure_logging()
    config, skill_name = build_config(args)
    env = setup_environment(config, [build_skill(name, config) for name in config["skills"]])
    policy = build_policy(args, config, skill_name, env)

    successes = 0
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed if episode == 0 else None, options={"skill": skill_name})
            setup = ", ".join(f"{grounded} ({steps} steps)" for grounded, steps in info["setup"]) or "none"
            print(f"=== episode {episode} target={info['grounded_action_model']} setup: {setup}")

            policy.reset()
            episode_return = 0.0
            for t in range(config["max_steps_per_episode"]):
                obs, reward, terminated, truncated, info = env.step(policy(obs, info, info["grounded_action_model"]))
                episode_return += reward
                if args.render:
                    time.sleep(args.sleep)
                if terminated or truncated:
                    break
            successes += int(info["is_success"])
            print(f"    success={bool(info['is_success'])} steps={t + 1} return={episode_return:.2f}")
    finally:
        env.close()
    print(f"success rate: {successes}/{args.episodes}")


if __name__ == "__main__":
    main()
