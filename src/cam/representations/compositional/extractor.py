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
    only in how the operator embedding is produced.

    The composed embedding is normalized to zero mean and unit L2 norm (operator_embedding). Without
    this, its scale is free to grow during training: in a short PPO run the tree architecture's
    embedding norm grew from 0.71 to 124 (a learnable vector stayed near 1.4), swamping the state and
    grounding inputs of the policy's MLP, and the tree runs never learned pickup. Unit norm also matches
    the scale of the random and text embeddings. `architecture` is TreeComposition, SlotComposition
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
        # No learnable gain or bias: a learnable gain could grow and reintroduce the scale explosion.
        self.normalize = nn.LayerNorm(output_dim, elementwise_affine=False)
        self.unit_norm_scale = output_dim**-0.5  # LayerNorm output has norm sqrt(output_dim); this makes it 1

    def operator_embedding(self, structure: torch.Tensor) -> torch.Tensor:
        """(B, layout.size) structure arrays -> (B, output_dim) composed embeddings with zero mean and
        unit L2 norm (up to LayerNorm's eps): what the policy sees (also used by scripts/review_runs.py)."""
        composed = self.composition(unpack(structure.float(), self.layout))
        return self.normalize(composed) * self.unit_norm_scale

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [observations[key].float().flatten(start_dim=1) for key in self.other_keys]
        parts.append(self.operator_embedding(observations[OPERATOR_EMBEDDING_KEY]))
        return torch.cat(parts, dim=1)
