"""Wrapper that builds the policy's observation: state, operator embedding, and grounding."""

import gymnasium as gym
import numpy as np

from cam.representations.grounding_encoder import GroundingEncoder
from cam.representations.operator_encoder import OperatorEncoder


class PolicyObservationWrapper(gym.Wrapper):
    """Builds the policy's observation: state, operator embedding, and grounding.

    Wraps an environment whose observations are {"observation": state vector} and
    whose info carries grounded_action_model, objects and object_features (e.g. a
    SkillEnvironment over the Fetch wrappers). Each observation becomes

        {"observation": state, "operator_embedding": encoder(lifted model), "grounding": grounding slots}

    Each distinct lifted model is encoded once and reused; the grounding is
    encoded every step. See docs/policy_inputs.md.
    """

    def __init__(self, env: gym.Env, operator_encoder: OperatorEncoder, grounding_encoder: GroundingEncoder):
        super().__init__(env)
        self.operator_encoder = operator_encoder
        self.grounding_encoder = grounding_encoder
        self.operator_embeddings: dict = {}
        self.observation_space = gym.spaces.Dict(
            {
                "observation": env.observation_space["observation"],
                "operator_embedding": gym.spaces.Box(-np.inf, np.inf, (operator_encoder.dim,), np.float32),
                "grounding": gym.spaces.Box(-np.inf, np.inf, (grounding_encoder.dim,), np.float32),
            }
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._policy_observation(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._policy_observation(obs, info), reward, terminated, truncated, info

    def _policy_observation(self, obs: dict, info: dict) -> dict:
        grounded = info["grounded_action_model"]
        lifted = grounded.lifted_model
        if lifted not in self.operator_embeddings:
            self.operator_embeddings[lifted] = self.operator_encoder.encode(lifted)
        return {
            "observation": obs["observation"],
            "operator_embedding": self.operator_embeddings[lifted],
            "grounding": self.grounding_encoder.encode(grounded, info["objects"], info["object_features"]),
        }
