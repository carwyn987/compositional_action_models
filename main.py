#!/usr/bin/env python3
"""Entry point: train skills, evaluate them, or both, from command-line options.

    # new run: train with evaluation during training (zero-shot, every --eval-interval steps, at the end)
    ./main.py --skills pickup putdown --algorithm sac --total-timesteps 200000 --reward shaped

    # resume a run (after an interruption, or to train further): --total-timesteps more steps
    ./main.py --run-directory outputs/pickup-putdown_sac_multi-hot_shaped_seed0

    # resume with an added (new or repaired) skill, until it reaches the success threshold
    ./main.py --run-directory outputs/<run> --skills pickup putdown stack --stop-skills stack

    # evaluate a run only (optionally other or held-out skills)
    ./main.py --mode evaluate --run-directory outputs/<run> --eval-skills stack --eval-episodes 50

A run directory holds config.json, metrics.json, model.zip, monitor.csv,
train.log and tensorboard/. Resuming reads config.json; options given
explicitly override it, except those that would change the policy's
observation or algorithm (see LOCKED_ON_RESUME).
"""

import argparse
import json
import logging
import shlex
import time
from pathlib import Path

import gymnasium as gym

from cam.domain.action_model_library.loader import SYMBOLIC_ACTION_MODEL_FORMATS
from cam.environments.fetch import fetch_multiblock_environment
from cam.experiments.environment_setup import (
    COMPOSITIONAL_ARCHITECTURES,
    OPERATOR_ENCODERS,
    compositional_layout,
    setup_environment,
)
from cam.logging_config import add_log_file, configure_logging
from cam.representations.text_backends import TEXT_BACKENDS
from cam.representations.text_operator_encoder import OPERATOR_TEXTS
from cam.skills.registry import SKILL_REGISTRY, build_skill
from cam.skills.skill import Skill

logger = logging.getLogger("cam.main")

MODES = ["train-evaluate", "train", "evaluate"]
# Options for one invocation only: not taken from a resumed run's config.json.
PER_INVOCATION = ["mode", "run_directory", "stop_skills", "patience", "eval_skills", "render"]
# Changing these would change the policy's observation or algorithm, so a resumed run keeps them.
LOCKED_ON_RESUME = [
    "algorithm", "environment_id", "num_blocks", "operator_encoder", "operator_embedding_dim", "max_operator_arity",
    "trainable_operator_embedding", "text_backend", "operator_text", "compositional_architecture",
    "component_embedding_dim", "max_operator_literals", "max_predicate_arity", "compositional_name", "slots",
    "slot_iterations",
    "symbolic_action_model_format",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode", choices=MODES, default="train-evaluate",
        help="train-evaluate: train with evaluation during training; train: no evaluation; "
        "evaluate: evaluate an existing run (--run-directory) without training",
    )
    parser.add_argument(
        "--run-directory",
        help="existing run: resumed by the training modes, evaluated by --mode evaluate; its config.json "
        "supplies all options not given explicitly",
    )

    skills = parser.add_argument_group("skills")
    skills.add_argument(
        "--skills", nargs="+", choices=sorted(SKILL_REGISTRY),
        help="skills to train (required for a new run); on resume, may add skills",
    )
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
    training.add_argument(
        "--total-timesteps", type=int, default=100_000,
        help="training environment steps; when resuming, steps to train in addition to the run's",
    )
    training.add_argument(
        "--max-steps-per-episode", type=int, default=100,
        help="episode time limit in actions (setup steps excluded); one Fetch action is 0.04 s of simulation, "
        "moving the gripper target by up to 5 cm per axis (scripted pickup needs about 27)",
    )
    training.add_argument("--output-dir", default="outputs", help="new runs are saved to <output-dir>/<run name>/")
    training.add_argument(
        "--save-interval", type=int, default=50_000,
        help="training steps between checkpoints, saved atomically (0: only at the end); "
        "the model is also saved when training ends, is interrupted, or crashes",
    )
    training.add_argument(
        "--save-replay-buffer", action="store_true",
        help="sac: also save the replay buffer with each checkpoint (can be large), so a resumed run keeps it",
    )
    training.add_argument("--seed", type=int, default=0)
    training.add_argument(
        "--operator-encoder", choices=sorted(OPERATOR_ENCODERS), default="multi-hot",
        help="how the lifted action model is embedded for the policy (docs/policy_inputs.md)",
    )
    training.add_argument(
        "--trainable-operator-embedding", action="store_true",
        help="pass the operator embedding through a learnable matrix initialised to the identity, trained with "
        "the policy: with --operator-encoder random, a learnable vector per operator starting at its random "
        "vector; with multi-hot, a learnable compositional (sum) embedding",
    )
    training.add_argument(
        "--operator-embedding-dim", type=int, default=128,
        help="size of the operator embedding, shared by all embedding methods so conditions are comparable "
        "(one-hot and multi-hot are zero-padded to it)",
    )
    training.add_argument(
        "--text-backend", choices=sorted(TEXT_BACKENDS), default="mock",
        help="--operator-encoder text: mock (offline character n-gram hashing) or openai (key from OPENAI_API_KEY, "
        "else the file in OPENAI_API_KEY_FILE, default ~/.secrets/openai_api_key)",
    )
    training.add_argument(
        "--operator-text", choices=sorted(OPERATOR_TEXTS), default="pddl",
        help="--operator-encoder text: embed the canonical PDDL, or a templated paragraph describing the operator",
    )
    training.add_argument(
        "--text-embedding-cache", default="outputs/text_embedding_cache.json",
        help="--operator-encoder text: cache of embeddings from non-mock backends, so each text is embedded once",
    )
    training.add_argument(
        "--max-operator-arity", type=int, default=3,
        help="most parameters any operator may have; fixes the multi-hot vocabulary and grounding slots",
    )
    training.add_argument(
        "--reward", choices=["sparse", "shaped"], default="sparse",
        help="shaped: Fetch shaped reward where one exists (pickup), sparse otherwise",
    )

    compositional = parser.add_argument_group(
        "compositional embedding (--operator-encoder compositional)",
        "The operator's components (types, variables, predicates, literals, name) have learnable embeddings, "
        "composed inside the policy into the operator embedding (--operator-embedding-dim) and trained with "
        "it. See src/cam/representations/compositional/.",
    )
    compositional.add_argument(
        "--compositional-architecture", choices=sorted(COMPOSITIONAL_ARCHITECTURES), default="tree",
        help="tree: learned NOT / AND / PRE / EFF / OPERATOR functions following the PDDL syntax; slots: slot "
        "attention over component tokens; geometric: components on a grid, read by a CNN and slot attention",
    )
    compositional.add_argument(
        "--component-embedding-dim", type=int, default=32, help="size of every component embedding",
    )
    compositional.add_argument(
        "--max-operator-literals", type=int, default=16, help="most preconditions + effects any operator may have",
    )
    compositional.add_argument(
        "--max-predicate-arity", type=int, default=4,
        help="most arguments any predicate may have (fixed, so modified operators fit the same layout)",
    )
    compositional.add_argument(
        "--compositional-name", choices=["text", "none"], default="text",
        help="include the operator name, text-embedded with --text-backend, or leave it out",
    )
    compositional.add_argument("--slots", type=int, default=4, help="slots (slots and geometric architectures)")
    compositional.add_argument(
        "--slot-iterations", type=int, default=3, help="slot attention rounds (slots and geometric architectures)",
    )

    evaluation = parser.add_argument_group(
        "evaluation",
        "During training (train-evaluate), evaluations run at the start (zero-shot), every --eval-interval steps "
        "and at the end, deterministic and stochastic, on a separate environment; results go to TensorBoard and "
        "<run>/metrics.json. Each costs about 2 x eval-episodes x skills episodes, so start with an infrequent "
        "interval and tune.",
    )
    evaluation.add_argument("--eval-interval", type=int, default=50_000, help="training steps between evaluations")
    evaluation.add_argument("--eval-episodes", type=int, default=20, help="episodes per skill per evaluation")
    evaluation.add_argument(
        "--success-threshold", type=float, default=0.95,
        help="success rate (over --eval-episodes) for the ..._to_threshold metrics and --stop-skills",
    )
    evaluation.add_argument(
        "--stop-skills", nargs="+", default=[], choices=sorted(SKILL_REGISTRY),
        help="stop training once all of these reach --success-threshold (deterministic evaluation)",
    )
    evaluation.add_argument(
        "--patience", type=int, default=0,
        help="stop training after this many evaluations without improvement of the mean success rate (0: off)",
    )
    evaluation.add_argument(
        "--eval-skills", nargs="+", choices=sorted(SKILL_REGISTRY),
        help="--mode evaluate: skills to evaluate (default: the run's skills); may include untrained skills",
    )
    return parser


def explicit_options(argv: list[str] | None) -> dict:
    """Only the options given on the command line (every default suppressed, including declared ones)."""
    parser = build_parser()
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    return vars(parser.parse_args(argv))


def parse_args(argv: list[str] | None = None) -> dict:
    """The run config: defaults, then the run's saved config.json (with --run-directory; except the
    PER_INVOCATION options), then explicit options."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = vars(args)
    if args.run_directory:
        config_path = Path(args.run_directory) / "config.json"
        if not config_path.exists():
            parser.error(f"{args.run_directory} has no config.json")
        saved = {key: value for key, value in json.loads(config_path.read_text()).items() if key not in PER_INVOCATION}
        explicit = explicit_options(argv)
        changed = [key for key in LOCKED_ON_RESUME if key in explicit and explicit[key] != saved.get(key)]
        if changed:
            parser.error(f"cannot change {changed} of an existing run (they define its observation and algorithm)")
        config = config | saved | explicit
    elif args.mode == "evaluate":
        parser.error("--mode evaluate needs --run-directory")
    if not config["skills"]:
        parser.error("--skills is required for a new run")
    if config["num_blocks"] is not None and config["environment_id"] != fetch_multiblock_environment.ENVIRONMENT_ID:
        parser.error(f"--num-blocks applies only to --environment-id {fetch_multiblock_environment.ENVIRONMENT_ID}")
    if config["operator_encoder"] == "compositional" and config["trainable_operator_embedding"]:
        parser.error("compositional embeddings are always trained with the policy; drop --trainable-operator-embedding")
    unknown_stop_skills = [skill for skill in config["stop_skills"] if skill not in config["skills"]]
    if unknown_stop_skills:
        parser.error(f"--stop-skills {unknown_stop_skills} are not among --skills {config['skills']}")
    return config


def run_name(config: dict) -> str:
    """e.g. pickup-putdown_sac_multi-hot_sparse_seed0, or ..._random-trainable_... with a trainable embedding"""
    encoder = config["operator_encoder"]
    if encoder == "text":
        encoder += f"-{config['text_backend']}-{config['operator_text']}"
    elif encoder == "compositional":
        encoder += f"-{config['compositional_architecture']}"
    if config.get("trainable_operator_embedding"):
        encoder += "-trainable"
    return "_".join(["-".join(config["skills"]), config["algorithm"], encoder, config["reward"], f"seed{config['seed']}"])


def policy_kwargs(config: dict) -> dict | None:
    """Policy options for a new model: the features extractor of a compositional or trainable operator
    embedding, shared by actor and critic (for SAC, then trained through the critic loss)."""
    if config["operator_encoder"] == "compositional":
        from cam.representations.compositional.extractor import CompositionalPolicyFeaturesExtractor

        architecture = config["compositional_architecture"]
        return {
            "features_extractor_class": CompositionalPolicyFeaturesExtractor,
            "features_extractor_kwargs": {
                "architecture": COMPOSITIONAL_ARCHITECTURES[architecture],
                "layout": compositional_layout(config),
                "component_dim": config["component_embedding_dim"],
                "output_dim": config["operator_embedding_dim"],
                "architecture_kwargs": (
                    {} if architecture == "tree"
                    else {"num_slots": config["slots"], "slot_iterations": config["slot_iterations"]}
                ),
            },
            "share_features_extractor": True,
        }
    if not config.get("trainable_operator_embedding"):
        return None
    from cam.policies.feature_extractors import TrainableOperatorEmbeddingExtractor

    return {"features_extractor_class": TrainableOperatorEmbeddingExtractor, "share_features_extractor": True}


def resume_command(run_directory: Path) -> str:
    return f"./main.py --run-directory {shlex.quote(str(run_directory))}"


def train(config: dict, env: gym.Env, skills: list[Skill], run_directory: Path, resume: bool) -> None:
    """Train with Stable-Baselines3 (resuming the run in run_directory if resume), with evaluation
    during training in train-evaluate mode."""
    from cam.evaluation.metrics_callback import MetricsCallback
    from cam.training.checkpointing import atomic_write_text
    from cam.training.stable_baselines3_trainer import OperatorEmbeddingMagnitudeCallback, train_stable_baselines3

    atomic_write_text(run_directory / "config.json", json.dumps(config, indent=2))
    callbacks, on_save, eval_env = [], [], None
    if config["trainable_operator_embedding"] or config["operator_encoder"] == "compositional":
        # learned operator embeddings: log their magnitude (L2 norm) per skill to TensorBoard
        operator_inputs = {skill.name: env.operator_encoder.encode(skill.symbolic_action_model) for skill in skills}
        callbacks.append(OperatorEmbeddingMagnitudeCallback(operator_inputs))
    if config["mode"] == "train-evaluate":
        metrics_path = run_directory / "metrics.json"
        eval_env = setup_environment(config | {"render": False}, skills)
        metrics_callback = MetricsCallback(
            eval_env,
            [skill.name for skill in skills],
            config["eval_interval"],
            config["eval_episodes"],
            run_directory,
            success_threshold=config["success_threshold"],
            seed=config["seed"] + 10_000,  # evaluation scenes differ from training's
            stop_skills=config["stop_skills"],
            patience=config["patience"],
            metrics=json.loads(metrics_path.read_text()) if resume and metrics_path.exists() else None,
        )
        callbacks.append(metrics_callback)
        on_save.append(metrics_callback.write_metrics)
    try:
        train_stable_baselines3(
            env,
            config["algorithm"],
            config["total_timesteps"],
            run_directory,
            seed=config["seed"],
            callbacks=callbacks,
            save_interval=config["save_interval"],
            resume=resume,
            save_replay_buffer=config["save_replay_buffer"],
            on_save=on_save,
            policy_kwargs=policy_kwargs(config),
        )
    finally:
        if eval_env is not None:
            eval_env.close()


def evaluate_run(config: dict, run_directory: Path) -> None:
    """Evaluate the run's model (deterministic and stochastic) on --eval-skills, without training;
    writes <run>/evaluation_<time>.json."""
    from cam.evaluation.evaluation import evaluate
    from cam.policies.stable_baselines3_policy import ALGORITHMS, StableBaselines3Policy
    from cam.training.checkpointing import atomic_write_text

    eval_skills = config["eval_skills"] or config["skills"]
    environment_skills = list(config["skills"]) + [skill for skill in eval_skills if skill not in config["skills"]]
    env = setup_environment(config, [build_skill(name, config) for name in environment_skills])
    model = ALGORITHMS[config["algorithm"]].load(run_directory / "model.zip")
    try:
        results = {
            mode: evaluate(
                env, StableBaselines3Policy(model, deterministic), eval_skills, config["eval_episodes"],
                seed=config["seed"] + 10_000,
            )
            for mode, deterministic in (("deterministic", True), ("stochastic", False))
        }
    finally:
        env.close()
    for mode, result in results.items():
        for skill, evaluation in result.skills.items():
            logger.info(
                "evaluation at step %d %s %s: success %.2f ± %.2f, return %.2f ± %.2f, length %.1f ± %.1f (%d episodes)",
                model.num_timesteps, mode, skill, evaluation.success_rate, evaluation.success_rate_std,
                evaluation.mean_return, evaluation.return_std,
                evaluation.mean_episode_length, evaluation.episode_length_std, evaluation.episodes,
            )
    path = run_directory / f"evaluation_{time.strftime('%Y%m%d-%H%M%S')}.json"
    atomic_write_text(path, json.dumps({
        "step": model.num_timesteps,
        "episodes_per_skill": config["eval_episodes"],
        "results": {mode: result.to_dict() for mode, result in results.items()},
    }, indent=2))
    logger.info("wrote %s", path)


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    configure_logging()
    resume = config["run_directory"] is not None
    run_directory = Path(config["run_directory"]) if resume else Path(config["output_dir"]) / run_name(config)
    if config["mode"] == "evaluate":
        evaluate_run(config, run_directory)
        return
    if resume and not (run_directory / "model.zip").exists():
        raise SystemExit(f"{run_directory} has no model.zip to resume")

    run_directory.mkdir(parents=True, exist_ok=True)
    config["run_directory"] = str(run_directory)
    add_log_file(run_directory / "train.log")
    logger.info("%s run %s with config:\n%s", "resuming" if resume else "starting", run_directory,
                json.dumps(config, indent=2))
    logger.info("to resume this run if it stops: %s", resume_command(run_directory))

    skills = [build_skill(skill_name, config) for skill_name in config["skills"]]
    env = setup_environment(config, skills)
    try:
        train(config, env, skills, run_directory, resume)
    finally:
        env.close()
        logger.info("to resume or continue this run: %s", resume_command(run_directory))


if __name__ == "__main__":
    main()
