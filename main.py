#!/usr/bin/env python3
"""Entry point: build skills and an environment from command-line options, then train.

    ./main.py --skills pickup putdown --algorithm sac --total-timesteps 200000 --reward shaped
    ./main.py --skills pickup --algorithm ppo --num-blocks 3
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym

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
    training.add_argument("--algorithm", choices=["sac", "ppo"], default="sac", help="Stable-Baselines3 algorithm")
    training.add_argument("--total-timesteps", type=int, default=100_000, help="training environment steps")
    training.add_argument(
        "--max-steps-per-episode", type=int, default=100,
        help="episode time limit in actions (setup steps excluded); one Fetch action is 0.04 s of simulation, "
        "moving the gripper target by up to 5 cm per axis (scripted pickup needs about 27)",
    )
    training.add_argument("--output-dir", default="outputs", help="runs are saved to <output-dir>/<run name>/")
    training.add_argument(
        "--save-interval", type=int, default=50_000,
        help="training steps between model.zip checkpoints, saved atomically (0: only at the end); "
        "the model is also saved when training ends, is interrupted, or crashes",
    )

    evaluation = parser.add_argument_group(
        "evaluation during training",
        "Evaluations run at the start (zero-shot), every --eval-interval steps and at the end, deterministic and "
        "stochastic, on a separate environment; results go to TensorBoard and <run>/metrics.json. Each costs about "
        "2 x eval-episodes x skills episodes, so start with an infrequent interval and tune.",
    )
    evaluation.add_argument("--eval-interval", type=int, default=50_000, help="training steps between evaluations; 0 disables")
    evaluation.add_argument("--eval-episodes", type=int, default=20, help="episodes per skill per evaluation")
    evaluation.add_argument(
        "--success-threshold", type=float, default=0.95,
        help="success rate (over --eval-episodes) for steps_to_threshold",
    )
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


def train(config: dict, env: gym.Env, skills: list[Skill]) -> None:
    """Train a policy with Stable-Baselines3.

    The run directory also gets config.json, the full training config, so the
    same environment can be rebuilt to evaluate the model (scripts/run_skill.py).
    """
    from cam.training.checkpointing import atomic_write_text
    from cam.training.stable_baselines3_trainer import train_stable_baselines3

    run_directory = Path(config["output_dir"]) / run_name(config)
    run_directory.mkdir(parents=True, exist_ok=True)
    atomic_write_text(run_directory / "config.json", json.dumps(config, indent=2))  # lets run_skill.py rebuild the env
    callbacks, eval_env = [], None
    if config["eval_interval"] > 0:
        from cam.evaluation.metrics_callback import MetricsCallback

        eval_env = setup_environment(config | {"render": False}, skills)
        callbacks.append(
            MetricsCallback(
                eval_env,
                [skill.name for skill in skills],
                config["eval_interval"],
                config["eval_episodes"],
                run_directory,
                success_threshold=config["success_threshold"],
                seed=config["seed"] + 10_000,  # evaluation scenes differ from training's
            )
        )
    try:
        train_stable_baselines3(
            env,
            config["algorithm"],
            config["total_timesteps"],
            run_directory,
            seed=config["seed"],
            callbacks=callbacks,
            save_interval=config["save_interval"],
        )
    finally:
        if eval_env is not None:
            eval_env.close()


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
