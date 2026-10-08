"""Composing the embedding in the policy network: the Stable-Baselines3 features extractor (see __init__.py)."""

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from cam.policies.feature_extractors import OPERATOR_EMBEDDING_KEY
from cam.representations.compositional.structure import OperatorLayout, unpack


class CompositionalOperatorExtractor(BaseFeaturesExtractor):
    """features = [grounding, observation, composition(operator structure)].

    The other observation parts pass through unchanged and the composed operator embedding (size
    output_dim) is appended last: the same layout as the other embedding conditions, so they differ
    only in how the operator embedding is produced. `architecture` is TreeComposition, SlotComposition
    or GeometricComposition, built with (layout, component_dim, output_dim, **architecture_kwargs).
    These settings are saved in model.zip, so a loaded model rebuilds the same network.
    """

    def __init__(self, observation_space: gym.spaces.Dict, architecture: type[nn.Module], layout: OperatorLayout,
                 component_dim: int, output_dim: int, architecture_kwargs: dict | None = None):
        self.other_keys = sorted(key for key in observation_space.spaces if key != OPERATOR_EMBEDDING_KEY)
        other_dim = sum(int(observation_space[key].shape[0]) for key in self.other_keys)
        super().__init__(observation_space, features_dim=other_dim + output_dim)
        self.layout = layout
        self.composition = architecture(layout, component_dim, output_dim, **(architecture_kwargs or {}))

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [observations[key].float().flatten(start_dim=1) for key in self.other_keys]
        parts.append(self.composition(unpack(observations[OPERATOR_EMBEDDING_KEY].float(), self.layout)))
        return torch.cat(parts, dim=1)
