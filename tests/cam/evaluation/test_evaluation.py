import gymnasium as gym
import numpy as np
import pytest

from cam.evaluation.evaluation import EvaluationResult, SkillEvaluation, evaluate
from cam.policies.policy import Policy


class CountdownEnv(gym.Env):
    """Episodes of a chosen skill; "easy" succeeds on step 1, "hard" never does and stops after 3 steps.

    Records the reset options and seeds it receives."""

    observation_space = gym.spaces.Box(0, 1, (1,))
    action_space = gym.spaces.Box(-1, 1, (1,))

    def __init__(self):
        self.resets = []

    def reset(self, *, seed=None, options=None):
        self.resets.append((seed, options["skill"]))
        self.skill, self.t = options["skill"], 0
        return np.zeros(1, np.float32), {"grounded_action_model": None}

    def step(self, action):
        self.t += 1
        success = self.skill == "easy"
        info = {"grounded_action_model": None, "is_success": float(success)}
        return np.zeros(1, np.float32), 1.0, success, self.t >= 3, info


class ZeroPolicy(Policy):
    def __call__(self, obs, info, grounded_action_model):
        return np.zeros(1, np.float32)


@pytest.mark.unit
def test_evaluate_summarises_each_skill():
    """Per skill: success rate, return and episode length means and stds over its episodes."""
    result = evaluate(CountdownEnv(), ZeroPolicy(), ["easy", "hard"], episodes_per_skill=4)
    assert result.skills["easy"] == SkillEvaluation(4, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    assert result.skills["hard"] == SkillEvaluation(4, 0.0, 0.0, 3.0, 0.0, 3.0, 0.0)


@pytest.mark.unit
def test_spread_is_the_std_over_episodes():
    """success_rate_std is the population std of the 0/1 outcomes."""
    evaluation = SkillEvaluation.from_episodes([1, 0, 1, 0], [2.0, 0.0, 2.0, 0.0], [1, 3, 1, 3])
    assert (evaluation.success_rate, evaluation.success_rate_std) == (0.5, 0.5)
    assert (evaluation.mean_return, evaluation.return_std) == (1.0, 1.0)
    assert (evaluation.mean_episode_length, evaluation.episode_length_std) == (2.0, 1.0)


@pytest.mark.unit
def test_episodes_select_the_skill_and_only_the_first_reset_is_seeded():
    """Each episode resets with options={"skill": name}; the seed goes to the first reset only."""
    env = CountdownEnv()
    evaluate(env, ZeroPolicy(), ["easy", "hard"], episodes_per_skill=2, seed=7)
    assert env.resets == [(7, "easy"), (None, "easy"), (None, "hard"), (None, "hard")]


@pytest.mark.unit
def test_result_round_trips_through_a_dict():
    """to_dict / from_dict (used for metrics.json) preserve every field."""
    result = evaluate(CountdownEnv(), ZeroPolicy(), ["easy", "hard"], episodes_per_skill=2)
    assert EvaluationResult.from_dict(result.to_dict()) == result
