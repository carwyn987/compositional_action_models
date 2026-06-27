"""Train ONE policy on several symbolic Fetch skills at once.

A single Stable-Baselines3 model learns every requested skill (e.g. pickup and
putdown). The skills share the same observation/action space; the *only* signal
that tells the policy which skill to perform is its symbolic embedding. Each
episode the environment samples one skill and serves that skill's reward,
termination, and (via a one-hot id) embedding.

Two embedding modes, selected with ``--embed-trainable``:

* frozen (default) -- the per-skill embedding table is initialised from the
  pretrained embeddings and held fixed.
* trainable        -- the same table is an optimisable policy parameter and is
  updated by RL (and BC). The learned embedding is NOT written back to the
  pretrained embedding source; it lives only inside the saved policy. How the
  embeddings move over training is recorded to an embedding log (see
  ``--embed-log-dir``) for later visualisation with ``plot_embeddings.py``.

Because the embedding is the sole differentiator, the embedder must produce a
distinct vector per skill -- enforced at startup. (Vanilla / no-embedding cannot
do multi-skill training and is intentionally unsupported here.)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
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
    SymbolicActionModel,
    add_embedder_cli_args,
    build_embedder,
    embed_tag,
    embedder_config_from_args,
    model_basename,
)
from symb_model_embeddings.trainable_embedding import (
    SkillEmbeddingExtractor,
    SkillOneHotWrapper,
)


class MultiSkillEnv(gym.Env):
    """Route each episode to one of several single-skill envs.

    All sub-environments must share an observation and action space (they differ
    only by skill reward/termination and by the one-hot skill id their
    observation carries). On every ``reset`` a skill is selected and all
    subsequent steps go to that skill's env until the next reset.
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


class EmbeddingLogger(BaseCallback):
    """Snapshot the policy's embedding table over the course of training."""

    def __init__(self, extractor: SkillEmbeddingExtractor, record_freq: int) -> None:
        super().__init__()
        self.extractor = extractor
        self.record_freq = max(1, int(record_freq))
        self.snapshots: list[dict] = []
        self._last = -(10**9)

    def _snap(self) -> None:
        self.snapshots.append(
            {
                "step": int(self.num_timesteps),
                "embeddings": self.extractor.current_embeddings().tolist(),
            }
        )
        self._last = self.num_timesteps

    def _on_training_start(self) -> None:
        self._snap()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.record_freq:
            self._snap()
        return True

    def _on_training_end(self) -> None:
        self._snap()


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
    parser.add_argument(
        "--embed-log-dir",
        type=Path,
        default=Path("embedding_logs"),
        help="Where to write the embedding-trajectory log (for plot_embeddings.py).",
    )
    parser.add_argument(
        "--embed-record-freq",
        type=int,
        default=0,
        help="Timesteps between embedding snapshots (0 = auto, ~20 per run).",
    )
    parser.add_argument("--teacher", choices=["bc", "off"], default="bc")
    parser.add_argument("--teacher-rollouts", type=int, default=8)
    parser.add_argument("--teacher-gradient-steps", type=int, default=500)
    parser.add_argument("--teacher-batch-size", type=int, default=256)
    parser.add_argument("--teacher-lr", type=float, default=1e-4)
    add_embedder_cli_args(parser)
    return parser.parse_args()


def build_initial_embeddings(
    skills: list[str], args: argparse.Namespace
) -> np.ndarray:
    """Pretrained per-skill embedding matrix used to initialise the table.

    Built with a frozen embedder regardless of ``--embed-trainable`` (the
    trainable flag controls the *policy* parameter, not how the initial values
    are produced).
    """
    config = replace(embedder_config_from_args(args), trainable=False)
    embedder = build_embedder(config)
    vectors = []
    for name in skills:
        spec = require_skill(name)
        action_model = SymbolicActionModel(name=spec.name, description=spec.action_model)
        vectors.append(np.asarray(embedder.embed(action_model), dtype=np.float32))
    matrix = np.stack(vectors, axis=0)
    _assert_distinct_embeddings(skills, matrix)
    return matrix


def _assert_distinct_embeddings(skills: list[str], matrix: np.ndarray) -> None:
    """Fail loudly if any two skills start from the same embedding vector.

    With identical embeddings the shared policy gets no signal distinguishing the
    skills, which defeats the point of multi-skill training.
    """
    for i in range(len(skills)):
        for j in range(i + 1, len(skills)):
            if np.allclose(matrix[i], matrix[j]):
                raise ValueError(
                    f"Skills {skills[i]!r} and {skills[j]!r} produce identical "
                    "embeddings, so the shared policy cannot tell them apart. Use "
                    "an embedder that differs per skill."
                )


def build_skill_envs(
    skills: list[str], render_mode: str | None, seed: int
) -> dict[str, gym.Env]:
    """One env per skill, each tagged with its one-hot skill id."""
    n = len(skills)
    return {
        name: SkillOneHotWrapper(
            make_skill_env(name, render_mode=render_mode, seed=seed + i),
            skill_index=i,
            num_skills=n,
        )
        for i, name in enumerate(skills)
    }


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


def _scheme_label(config) -> str:
    if config.kind == "mock":
        return "mock"
    return "name" if config.source == "name" else "full"


def _find_embedding_extractor(model) -> SkillEmbeddingExtractor:
    for module in model.policy.modules():
        if isinstance(module, SkillEmbeddingExtractor):
            return module
    raise RuntimeError("SkillEmbeddingExtractor not found in the policy")


def write_embedding_log(
    path: Path,
    *,
    basename: str,
    skills: list[str],
    config,
    args: argparse.Namespace,
    initial: np.ndarray,
    snapshots: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "basename": basename,
        "skills": skills,
        "scheme": _scheme_label(config),
        "trainable": bool(config.trainable),
        "embedder": {
            "kind": config.kind,
            "source": config.source,
            "backend": config.backend,
            "size": config.size,
        },
        "algo": args.algo,
        "seed": args.seed,
        "dim": int(initial.shape[1]),
        "initial": initial.astype(float).tolist(),
        "snapshots": snapshots,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote embedding log to {path}")


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.logdir.mkdir(parents=True, exist_ok=True)

    config = embedder_config_from_args(args)
    skill_token = "multi-" + "-".join(args.skills)
    basename = model_basename(
        skill_token, args.algo, embed_tag(config), args.teacher, args.seed
    )

    init_matrix = build_initial_embeddings(args.skills, args)
    render_mode = "human" if args.render else None
    envs = build_skill_envs(args.skills, render_mode, args.seed)
    multi_env = MultiSkillEnv(envs, seed=args.seed, strategy=args.sampling)
    env = Monitor(multi_env, filename=str(args.logdir / f"{basename}.monitor.csv"))

    policy_kwargs = dict(
        features_extractor_class=SkillEmbeddingExtractor,
        features_extractor_kwargs=dict(
            init_embeddings=init_matrix, trainable=config.trainable
        ),
        share_features_extractor=True,
    )

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
            policy_kwargs=policy_kwargs,
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
            policy_kwargs=policy_kwargs,
        )

    if args.teacher == "bc":
        warm_start_multiskill(model, envs, args)
    else:
        print("Teacher warm-start disabled")

    extractor = _find_embedding_extractor(model)
    record_freq = args.embed_record_freq or max(1, args.timesteps // 20)
    logger = EmbeddingLogger(extractor, record_freq)

    mode = "trainable" if config.trainable else "frozen"
    print(f"Training multi-skill model [{mode}] on: {', '.join(args.skills)}")
    model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=logger)

    out_path = args.models_dir / basename
    model.save(out_path)
    write_embedding_log(
        args.embed_log_dir / f"{basename}.json",
        basename=basename,
        skills=args.skills,
        config=config,
        args=args,
        initial=init_matrix,
        snapshots=logger.snapshots,
    )
    env.close()
    print(f"Saved multi-skill model ({mode}) to {out_path}.zip")


if __name__ == "__main__":
    main()
