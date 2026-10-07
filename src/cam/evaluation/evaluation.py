"""Evaluate a policy on skills: success rate, return and episode length, with their spread.

evaluate() runs a number of episodes per skill and summarises them. It is used
for zero-shot evaluation (before training), periodically during training (the
learning curves), and after training. The policy decides how actions are chosen
(e.g. deterministic or stochastic); callers that evaluate both store the two
results separately.
"""

from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np

from cam.policies.policy import Policy


@dataclass(frozen=True)
class SkillEvaluation:
    """Summary of one skill's evaluation episodes. Standard deviations are over episodes
    (population, ddof=0); success_rate_std is the std of the per-episode 0/1 outcomes."""

    episodes: int
    success_rate: float
    success_rate_std: float
    mean_return: float
    return_std: float
    mean_episode_length: float
    episode_length_std: float

    @classmethod
    def from_episodes(cls, successes: list[float], returns: list[float], lengths: list[int]) -> "SkillEvaluation":
        return cls(
            episodes=len(successes),
            success_rate=float(np.mean(successes)),
            success_rate_std=float(np.std(successes)),
            mean_return=float(np.mean(returns)),
            return_std=float(np.std(returns)),
            mean_episode_length=float(np.mean(lengths)),
            episode_length_std=float(np.std(lengths)),
        )


@dataclass(frozen=True)
class EvaluationResult:
    """Per-skill evaluation summaries, keyed by skill name."""

    skills: dict[str, SkillEvaluation]

    def to_dict(self) -> dict:
        return {name: asdict(evaluation) for name, evaluation in self.skills.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationResult":
        return cls({name: SkillEvaluation(**fields) for name, fields in data.items()})


def evaluate(
    env: gym.Env,
    policy: Policy,
    skill_names: list[str],
    episodes_per_skill: int,
    seed: int | None = None,
) -> EvaluationResult:
    """Run episodes_per_skill episodes of each skill and summarise them.

    env must run one skill per episode via reset(options={"skill": name}), end
    episodes itself (e.g. TimeLimit), and report info["is_success"] (e.g. a
    SkillEnvironment stack). seed seeds the first reset, so evaluations with the
    same seed start from the same scenes and are comparable across checkpoints.
    """
    results = {}
    for skill_index, skill_name in enumerate(skill_names):
        successes, returns, lengths = [], [], []
        for episode in range(episodes_per_skill):
            first = skill_index == 0 and episode == 0
            obs, info = env.reset(seed=seed if first else None, options={"skill": skill_name})
            policy.reset()
            episode_return, length, terminated, truncated = 0.0, 0, False, False
            while not (terminated or truncated):
                obs, reward, terminated, truncated, info = env.step(policy(obs, info, info["grounded_action_model"]))
                episode_return += float(reward)
                length += 1
            successes.append(float(info["is_success"]))
            returns.append(episode_return)
            lengths.append(length)
        results[skill_name] = SkillEvaluation.from_episodes(successes, returns, lengths)
    return EvaluationResult(results)
