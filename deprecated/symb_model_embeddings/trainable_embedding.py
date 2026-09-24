"""Learnable symbolic embeddings for multi-skill policies.

In the frozen schemes the embedding vector is part of the *observation* (see
``obs_symb_wrapper.EnvWrapper``), so it is data and cannot receive gradients. To
let the embedding be *trained*, the vector instead has to live inside the policy
as a parameter. This module provides the two pieces that make that possible:

* :class:`SkillOneHotWrapper` -- puts a one-hot *skill id* in the observation
  (instead of the embedding vector). It tells the policy which skill the current
  episode is, without baking in the embedding values.
* :class:`SkillEmbeddingExtractor` -- an SB3 features extractor that owns a table
  of per-skill embedding vectors (initialised from the pretrained embeddings) and
  looks the right one up via the one-hot id. The table is an ``nn.Parameter``
  whose ``requires_grad`` is the ``trainable`` flag, so the same class serves
  both the frozen and the trainable modes.

Kept out of the package ``__init__`` on purpose: importing it pulls in torch, and
the lightweight consumers (e.g. the ``run_naming`` CLI the bash drivers call)
should not pay that cost.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

SKILL_ID_KEY = "skill_onehot"


class SkillOneHotWrapper(gym.ObservationWrapper):
    """Append a fixed one-hot skill id to the observation.

    Unlike :class:`~symb_model_embeddings.obs_symb_wrapper.EnvWrapper`, this does
    not add the embedding vector itself -- the embedding lives in the policy (see
    :class:`SkillEmbeddingExtractor`). The id identifies which row of the policy's
    embedding table this skill uses.
    """

    def __init__(self, env: gym.Env, skill_index: int, num_skills: int) -> None:
        super().__init__(env)
        if not 0 <= skill_index < num_skills:
            raise ValueError(
                f"skill_index {skill_index} out of range for num_skills {num_skills}"
            )
        self.skill_index = int(skill_index)
        self.num_skills = int(num_skills)
        onehot = np.zeros(self.num_skills, dtype=np.float32)
        onehot[self.skill_index] = 1.0
        self._onehot = onehot

        spaces = dict(env.observation_space.spaces)
        spaces[SKILL_ID_KEY] = Box(
            low=0.0, high=1.0, shape=(self.num_skills,), dtype=np.float32
        )
        self.observation_space = Dict(spaces)

    def observation(self, obs: dict) -> dict:
        return obs | {SKILL_ID_KEY: self._onehot}


class SkillEmbeddingExtractor(BaseFeaturesExtractor):
    """Features extractor with a per-skill embedding table.

    The non-id observation entries are flattened and concatenated with the
    embedding selected by the one-hot skill id. Selection is done as a matrix
    multiply (``onehot @ table``) so gradients flow into the table when it is
    trainable.
    """

    def __init__(
        self,
        observation_space: Dict,
        init_embeddings: np.ndarray,
        trainable: bool = False,
    ) -> None:
        init = np.asarray(init_embeddings, dtype=np.float32)
        if init.ndim != 2:
            raise ValueError("init_embeddings must be a 2-D (num_skills, dim) array")
        _, dim = init.shape

        self._other_keys = [k for k in observation_space.spaces if k != SKILL_ID_KEY]
        base_dim = sum(
            int(np.prod(observation_space.spaces[k].shape)) for k in self._other_keys
        )
        super().__init__(observation_space, base_dim + dim)

        self.embed_dim = int(dim)
        # torch.tensor copies, so the parameter owns its storage and in-place
        # optimizer updates never mutate the caller's init array.
        self.embeddings = nn.Parameter(
            torch.tensor(init, dtype=torch.float32), requires_grad=bool(trainable)
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        onehot = observations[SKILL_ID_KEY].float()
        emb = onehot @ self.embeddings  # (batch, dim)
        parts = [
            observations[k].float().reshape(observations[k].shape[0], -1)
            for k in self._other_keys
        ]
        parts.append(emb)
        return torch.cat(parts, dim=1)

    def current_embeddings(self) -> np.ndarray:
        """Return a copy of the current embedding table as a numpy array."""
        return self.embeddings.detach().cpu().numpy().astype(np.float32).copy()
