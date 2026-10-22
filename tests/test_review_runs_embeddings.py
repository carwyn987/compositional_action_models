"""scripts/review_runs.py section 6 (embeddings): it finds a run's learned operator embedding for every algorithm."""

import json

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

import torch  # noqa: E402
from stable_baselines3 import PPO, SAC  # noqa: E402

from cam.policies.feature_extractors import features_extractors  # noqa: E402
from main import parse_args, policy_kwargs  # noqa: E402
from scripts.review_runs import section_embeddings  # noqa: E402


class OperatorEnv(gym.Env):
    """The policy observation's shape (an observation and an operator embedding of 256), with nothing to learn."""

    observation_space = gym.spaces.Dict({
        "observation": gym.spaces.Box(-1, 1, (1,)), "operator_embedding": gym.spaces.Box(-np.inf, np.inf, (256,)),
    })
    action_space = gym.spaces.Box(-1, 1, (1,))

    def reset(self, *, seed=None, options=None):
        return {key: np.zeros(space.shape, np.float32) for key, space in self.observation_space.items()}, {}

    def step(self, action):
        return self.reset()[0], 0.0, True, False, {}


@pytest.mark.unit
def test_section_6_reads_a_sac_runs_embedding_from_its_actor(tmp_path):
    """A SAC policy's own features_extractor is None (its extractors are in its actor and critics), so the lookup
    section 6 used found nothing for SAC runs (the control); features_extractors finds the actor's, and section 6
    reports the run's embeddings: W times each skill's multi-hot vector, here with W = 2 I (pickup has 16 features,
    so magnitude 2 x 4; putdown 10, so 2 x sqrt(10))."""
    config = parse_args(["--skills", "pickup", "putdown", "--algorithm", "sac", "--trainable-operator-embedding"])
    model = SAC("MultiInputPolicy", OperatorEnv(), policy_kwargs=policy_kwargs(config) | {"net_arch": [8]},
                device="cpu")
    policy = model.policy
    with torch.no_grad():
        policy.actor.features_extractor.operator_embedding.weight.mul_(2.0)
    model.save(tmp_path / "model.zip")
    (tmp_path / "config.json").write_text(json.dumps(config))

    old_lookup = policy.features_extractor if hasattr(policy, "features_extractor") else policy.actor.features_extractor
    assert old_lookup is None
    # the critic shares the actor's extractor, so it is listed once
    assert features_extractors(policy) == {
        "actor": policy.actor.features_extractor, "critic-target": policy.critic_target.features_extractor,
    }
    _, line = section_embeddings([tmp_path])
    assert line.startswith(f"   {tmp_path.name}: magnitudes pickup 8, putdown 6.32;")


@pytest.mark.unit
@pytest.mark.parametrize("share", [True, False])
def test_ppo_extractors_are_the_policy_and_value_functions(share):
    """PPO's policy (pi) extractor is its actor, its value function's (vf) the critic: listed once when shared."""
    config = parse_args(["--skills", "pickup", "--algorithm", "ppo", "--trainable-operator-embedding"])
    kwargs = policy_kwargs(config) | {"net_arch": [8], "share_features_extractor": share}
    policy = PPO("MultiInputPolicy", OperatorEnv(), policy_kwargs=kwargs, n_steps=64, batch_size=32, device="cpu").policy
    expected = {"actor": policy.pi_features_extractor} | ({} if share else {"critic": policy.vf_features_extractor})
    assert features_extractors(policy) == expected
