"""Train a policy with Stable-Baselines3 (SAC or PPO) on the policy observation.

The environment is the full stack from main.setup_environment, whose
observations are {"observation", "operator_embedding", "grounding"}; SB3's
MultiInputPolicy concatenates them and feeds an MLP. One model is trained on
all skills: SkillEnvironment chooses the episode's skill, and the operator
embedding and grounding tell the policy which skill and objects it is acting on.

Outputs, under <output_dir>/<run_name>/:
    model.zip          the SB3 model: saved every save_interval steps, at the end, and on interruption
    monitor.csv        per-episode return, length, is_success, skill
    tensorboard/       SB3 training curves, plus success_rate/<skill> (and eval_<mode>/<skill>/... with a
                       MetricsCallback, which also writes metrics.json)

Hyperparameters follow the original deprecated/train_skill.py (removed; see git commit 59054395).
"""

import json
import logging
from collections import deque
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from cam.policies.feature_extractors import OPERATOR_EMBEDDING_KEY, OperatorEmbeddingSACPolicy, features_extractors
from cam.training.checkpointing import (
    PeriodicSaveCallback,
    atomic_save_model,
    atomic_save_replay_buffer,
    atomic_write_text,
    sigint_ignored,
)
from cam.training.multi_task_sac import MultiTaskSAC

logger = logging.getLogger(__name__)

ALGORITHMS = {"sac": SAC, "mtsac": MultiTaskSAC, "ppo": PPO}
HYPERPARAMETERS = {
    "sac": dict(
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=1_000,
        batch_size=256,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
    ),
    "ppo": dict(learning_rate=3e-4, n_steps=1024, batch_size=256, gamma=0.95),
}
HYPERPARAMETERS["mtsac"] = HYPERPARAMETERS["sac"]
# PPO's Gaussian action noise is the same in every state. At Stable-Baselines3's default (standard deviation 1) a
# grasped block is let go within a few steps, since any positive gripper command opens the fingers: the scripted
# pickup with that noise on the gripper command alone succeeds 0 of 20 times (with 0.3, 20 of 20), so a policy
# learns that grasping fails. New PPO models start at exp(-1), about 0.37.
PPO_LOG_STD_INIT = -1.0


# After resuming an off-policy model without its replay buffer, the buffer starts empty, and Stable-Baselines3 updates
# from the first new step on batches drawn from the few transitions collected since: the critic fits them and the
# policy forgets the skills that are not among them. Instead the policy acts without updates for this many steps first.
RESUME_WARMUP_STEPS = 5_000


class ResumeWarmupCallback(BaseCallback):
    """No gradient updates for the first `steps` environment steps of this training invocation (the policy still acts,
    filling the replay buffer), then the model's usual gradient steps."""

    def __init__(self, steps: int):
        super().__init__()
        self.steps = steps
        self.gradient_steps: int | None = None  # the model's own, while the warm-up holds them at 0
        self.start = 0

    def _on_training_start(self) -> None:
        self.gradient_steps, self.start = self.model.gradient_steps, self.model.num_timesteps
        self.model.gradient_steps = 0

    def _on_step(self) -> bool:
        if self.model.num_timesteps - self.start >= self.steps:
            self._end_warmup()
        return True

    def _on_training_end(self) -> None:
        self._end_warmup()

    def _end_warmup(self) -> None:
        if self.gradient_steps is not None:
            self.model.gradient_steps, self.gradient_steps = self.gradient_steps, None


def remove_transitions(replay_buffer: DictReplayBuffer, operator_embeddings: list[np.ndarray]) -> int:
    """Removes every transition whose operator embedding (obs["operator_embedding"]) is one of operator_embeddings
    from replay_buffer, and returns how many it removed. The others move to the front, oldest first, so new
    transitions still replace the oldest. Only the arrays in memory change, never the file the buffer was loaded
    from."""
    if replay_buffer.n_envs != 1:
        raise ValueError("removing transitions needs a replay buffer of one environment")
    stored = replay_buffer.observations[OPERATOR_EMBEDDING_KEY][: replay_buffer.size(), 0]
    removed = np.zeros(len(stored), dtype=bool)
    for embedding in operator_embeddings:
        removed |= (stored == embedding).all(axis=1)
    if not removed.any():
        return 0
    oldest = replay_buffer.pos if replay_buffer.full else 0
    order = (np.arange(len(stored)) + oldest) % len(stored)
    kept = order[~removed[order]]
    for array in (*replay_buffer.observations.values(), *replay_buffer.next_observations.values(),
                  replay_buffer.actions, replay_buffer.rewards, replay_buffer.dones, replay_buffer.timeouts):
        array[: len(kept)] = array[kept]
    replay_buffer.pos, replay_buffer.full = len(kept), False
    return int(removed.sum())


def forget_replaced_skills(model: BaseAlgorithm, operator_embeddings: dict[str, np.ndarray]) -> None:
    """At a resume that replaces skills (a repair), removes what the model holds of them besides its network: their
    transitions in the replay buffer and, for MultiTaskSAC, their temperatures (reset to the initial one).
    operator_embeddings: each replaced skill's operator embedding, as the run being resumed encoded it.

    A replaced skill is no longer part of the task. Its transitions' rewards are for effects the skill no longer has,
    and with repair-hidden their inputs are exactly the repaired operator's, so the critic would get contradictory
    targets in the very states the repair changes until FIFO replacement flushed them; its temperature was tuned to a
    behaviour no longer wanted. SAC's one temperature is every skill's, and what a new skill gets too, so it stays."""
    names = ", ".join(operator_embeddings)
    replay_buffer = getattr(model, "replay_buffer", None)
    if replay_buffer is not None:
        removed = remove_transitions(replay_buffer, list(operator_embeddings.values()))
        logger.info("replaced skills %s: removed their %d transitions from the replay buffer, %d left",
                    names, removed, replay_buffer.size())
    if isinstance(model, MultiTaskSAC):
        slots = model.reset_temperatures(operator_embeddings.values())
        logger.info("replaced skills %s: temperature slots %s reset to %.3g",
                    names, slots, float(np.exp(model.initial_log_ent_coef)))
    elif replay_buffer is not None:
        logger.info("replaced skills %s: SAC's one temperature, every skill's, is kept", names)


class ProgressFileCallback(BaseCallback):
    """Writes this training invocation's progress to path as JSON, {"done": steps done, "total": steps it trains,
    "step": the model's step}, at the start, every `every` steps and at the end. The overnight runner reads it: the
    progress bar is drawn only on a terminal, so written to a run's log file it appears once, at the end."""

    def __init__(self, path: Path, total: int, every: int = 1000):
        super().__init__()
        self.path, self.total, self.every = path, total, every
        self.start = self.last = 0

    def _on_training_start(self) -> None:
        self.start = self.model.num_timesteps
        self._write()

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last >= self.every:
            self._write()
        return True

    def _on_training_end(self) -> None:
        self._write()

    def _write(self) -> None:
        self.last = self.num_timesteps
        atomic_write_text(self.path, json.dumps(
            {"done": self.num_timesteps - self.start, "total": self.total, "step": self.num_timesteps}))


class SuccessRateCallback(BaseCallback):
    """Logs each skill's success rate over its last `window` episodes as success_rate/<skill>.

    Reads info["skill"] and info["is_success"] from SkillEnvironment at the end of
    each episode. SB3 also logs rollout/success_rate across all skills.
    """

    def __init__(self, window: int = 100):
        super().__init__()
        self.window = window
        self.outcomes: dict[str, deque] = {}

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                outcomes = self.outcomes.setdefault(info["skill"], deque(maxlen=self.window))
                outcomes.append(float(info["is_success"]))
                self.logger.record(f"success_rate/{info['skill']}", float(np.mean(outcomes)))
        return True


def operator_embedding_magnitudes(extractor, operator_inputs: np.ndarray) -> dict[str, np.ndarray]:
    """Magnitudes (L2 norms, i.e. lengths) of the operator embeddings a features extractor computes from
    operator_inputs (one obs["operator_embedding"] per row), keyed by TensorBoard tag prefix:
        operator_embedding_magnitude                        what the policy sees
        operator_embedding_magnitude_before_normalization   compositional only: the composition's raw output
    Empty for an extractor without a learned operator embedding (the fixed embeddings never change)."""
    operator_embedding = getattr(extractor, "operator_embedding", None)
    if operator_embedding is None:
        return {}
    device = next(extractor.parameters()).device
    inputs = torch.as_tensor(operator_inputs, dtype=torch.float32, device=device)
    with torch.no_grad():
        magnitudes = {"operator_embedding_magnitude": operator_embedding(inputs).norm(dim=1)}
        if hasattr(extractor, "composed_embedding"):
            magnitudes["operator_embedding_magnitude_before_normalization"] = extractor.composed_embedding(inputs).norm(dim=1)
    return {tag: values.cpu().numpy() for tag, values in magnitudes.items()}


class OperatorEmbeddingMagnitudeCallback(BaseCallback):
    """Every log_interval steps, logs each skill's operator embedding magnitude (L2 norm, i.e. length) as
    the policy currently computes it, as operator_embedding_magnitude/<skill> (compositional runs also
    ..._before_normalization/<skill>; see operator_embedding_magnitudes). SAC's extractor is logged as
    .../actor/<skill>, and a critic's own extractor, when it does not share the actor's, as .../critic/<skill>.

    A large or growing magnitude swamps the other inputs of the policy's MLP (around 1): the unnormalized
    compositional tree embedding started at about 100 and reached hundreds. This shows such problems
    during training instead of only after it.
    """

    def __init__(self, operator_inputs: dict[str, np.ndarray], log_interval: int = 1000):
        super().__init__()
        self.skills = list(operator_inputs)
        self.inputs = np.stack([operator_inputs[skill] for skill in self.skills])
        self.log_interval = log_interval
        self.last_logged: int | None = None

    def _extractors(self) -> dict[str, object]:
        policy = self.model.policy
        if hasattr(policy, "actor"):
            return {f"{role}/": extractor for role, extractor in features_extractors(policy).items()
                    if role != "critic-target"}
        return {"": policy.features_extractor}

    def _on_step(self) -> bool:
        if self.last_logged is None or self.num_timesteps - self.last_logged >= self.log_interval:
            self.last_logged = self.num_timesteps
            for part, extractor in self._extractors().items():
                for tag, magnitudes in operator_embedding_magnitudes(extractor, self.inputs).items():
                    for skill, magnitude in zip(self.skills, magnitudes):
                        self.logger.record(f"{tag}/{part}{skill}", float(magnitude))
        return True


class CompositionWeightDecayCallback(BaseCallback):
    """Decoupled weight decay (as in AdamW) on the compositional operator embedding's parameters only.

    The extractor normalizes the composition's output, so nothing in the loss limits the composition's scale
    and its weights drift upward (its raw output grew from ~2.5 to ~2800 in 500k SAC steps), slowing how
    much each update changes what the policy sees. After every gradient update this multiplies each parameter
    under a features extractor's `composition` by (1 - learning_rate * weight_decay), pulling the scale back.

    Weight decay has no task-specific learning direction: it simply shrinks every parameter toward zero. The
    learning signal has to keep re-growing what it needs, so the scale settles where the two balance.
    Drawbacks: too strong a decay shrinks features the task needs (the composition underfits, operators
    blur together) and acts as a constant pull against learning; it shrinks every composition parameter
    alike, including component embeddings of rarely seen predicates, which get few gradient updates to
    restore them; and it adds a setting to tune (watch operator_embedding_magnitude_before_normalization).
    """

    def __init__(self, weight_decay: float):
        super().__init__()
        self.weight_decay = weight_decay
        self.updates_seen: int | None = None

    def _on_step(self) -> bool:
        updates = getattr(self.model, "_n_updates", 0)
        new = updates - (updates if self.updates_seen is None else self.updates_seen)
        self.updates_seen = updates
        if new > 0:
            factor = (1.0 - self.model.lr_schedule(self.model._current_progress_remaining) * self.weight_decay) ** new
            # the extractors a gradient step updates, each once: SAC's actor and critic share one (main.py's
            # policy_kwargs). The target critic's copy is not one: Polyak averaging, a moving average, moves its
            # weights a small fraction tau toward the critic's after each gradient update
            # (target = tau * critic + (1 - tau) * target)
            feature_extractors = [
                extractor for role, extractor in features_extractors(self.model.policy).items()
                if role != "critic-target"
            ]
            with torch.no_grad():
                for extractor in feature_extractors:
                    for parameter in getattr(extractor, "composition", torch.nn.Module()).parameters():
                        parameter.mul_(factor)
        return True


def train_stable_baselines3(
    env: gym.Env,
    algorithm: str,
    total_timesteps: int,
    run_directory: Path,
    seed: int = 0,
    hyperparameter_overrides: dict | None = None,
    callbacks: list[BaseCallback] | None = None,
    save_interval: int = 0,
    resume: bool = False,
    save_replay_buffer: bool = False,
    on_save: list[Callable[[], None]] | None = None,
    policy_kwargs: dict | None = None,
    operator_embedding_loss: str = "critic",
    device: str = "auto",
    resume_warmup_steps: int = RESUME_WARMUP_STEPS,
    replaced_skills: dict[str, np.ndarray] | None = None,
) -> BaseAlgorithm:
    """Train an SB3 model on env for total_timesteps environment steps and save it to run_directory.

    resume=True continues the model in run_directory/model.zip (and its replay_buffer.pkl, if
    saved) for total_timesteps more steps: the step count, TensorBoard curves and monitor.csv
    continue. callbacks run alongside SuccessRateCallback (e.g. a MetricsCallback).

    Saving (atomic) happens every save_interval steps (0: only at the end), at the end, and when
    training is interrupted (Ctrl+C) or raises, before the exception propagates. It writes
    model.zip, replay_buffer.pkl if save_replay_buffer (off-policy algorithms; can be large),
    and calls each on_save function (e.g. writing metrics).

    policy_kwargs go to a new model's policy (e.g. a features extractor); a resumed model keeps its own. A new SAC
    or mtsac model's policy is OperatorEmbeddingSACPolicy, whose features extractor, when the actor and critic share
    it, is trained by the loss operator_embedding_loss names ("critic" or "actor"). PPO's policy is Stable-Baselines3's
    own: its one loss, of actor and value function together, trains a shared extractor, so the option does not apply.
    device is the torch device for the policy ("auto": a CUDA GPU if available, else cpu).

    An off-policy model resumed without a saved replay buffer makes no gradient updates for its first
    resume_warmup_steps steps (ResumeWarmupCallback), so it does not train on a nearly empty buffer.

    replaced_skills (resume only; skill name -> operator embedding as the run being resumed encoded it): skills the
    resume replaces; their transitions are removed and their temperatures reset before training
    (forget_replaced_skills).
    """
    run_directory.mkdir(parents=True, exist_ok=True)
    model_path = run_directory / "model.zip"
    replay_buffer_path = run_directory / "replay_buffer.pkl"
    monitored_env = Monitor(
        env, filename=str(run_directory), info_keywords=("is_success", "skill"), override_existing=not resume
    )
    if resume:
        model = ALGORITHMS[algorithm].load(
            model_path, env=monitored_env, tensorboard_log=str(run_directory / "tensorboard"), device=device
        )
        warmup = []
        if hasattr(model, "replay_buffer"):
            if model.gradient_steps == 0:  # saved during an interrupted warm-up
                model.gradient_steps = HYPERPARAMETERS[algorithm]["gradient_steps"]
            if replay_buffer_path.exists():
                model.load_replay_buffer(replay_buffer_path)
                logger.info("loaded replay buffer from %s", replay_buffer_path)
            elif resume_warmup_steps > 0:
                warmup = [ResumeWarmupCallback(resume_warmup_steps)]
                logger.info("no saved replay buffer: no updates for the first %d steps", resume_warmup_steps)
        if replaced_skills:
            forget_replaced_skills(model, replaced_skills)
        logger.info("resuming %s from step %d", algorithm, model.num_timesteps)
    else:
        policy, options = (("MultiInputPolicy", {"log_std_init": PPO_LOG_STD_INIT}) if algorithm == "ppo"
                           else (OperatorEmbeddingSACPolicy, {"operator_embedding_loss": operator_embedding_loss}))
        model = ALGORITHMS[algorithm](
            policy,
            monitored_env,
            seed=seed,
            verbose=1,
            tensorboard_log=str(run_directory / "tensorboard"),
            policy_kwargs=options | (policy_kwargs or {}),
            device=device,
            **(HYPERPARAMETERS[algorithm] | (hyperparameter_overrides or {})),
        )

    def save() -> None:
        atomic_save_model(model, model_path)
        if save_replay_buffer and getattr(model, "replay_buffer", None) is not None:
            atomic_save_replay_buffer(model, replay_buffer_path)
        for save_more in on_save or []:
            save_more()

    all_callbacks = [SuccessRateCallback(), ProgressFileCallback(run_directory / "progress.json", total_timesteps),
                     *(warmup if resume else []), *(callbacks or [])]
    if save_interval > 0:
        all_callbacks.append(PeriodicSaveCallback(save_interval, save))
    logger.info("training %s for %d steps on %s; outputs in %s", algorithm, total_timesteps, model.device, run_directory)
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=all_callbacks,
            reset_num_timesteps=not resume,
            progress_bar=True,
        )
    except BaseException:
        with sigint_ignored():
            save()
        logger.warning("training stopped at step %d; saved model to %s", model.num_timesteps, model_path)
        raise
    save()
    logger.info("training finished at step %d; saved model to %s", model.num_timesteps, model_path)
    return model
