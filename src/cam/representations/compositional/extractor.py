"""Composing the embedding in the policy network: the Stable-Baselines3 features extractor (see __init__.py)."""

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from cam.policies.feature_extractors import OPERATOR_EMBEDDING_KEY
from cam.representations.compositional.structure import OperatorLayout, unpack


class CompositionalPolicyFeaturesExtractor(BaseFeaturesExtractor):
    """Policy side of the compositional condition: the first module of the policy network, where the
    operator embedding is composed and trained.

    Where it fits:
        environment side   StructuredOperatorEncoder (structure.py) writes the operator's structure array
                           into obs["operator_embedding"]; nothing there is learned
        policy side        this extractor (Stable-Baselines3's features_extractor_class, set in main.py)
                           unpacks that array, composes the component embeddings with the chosen
                           architecture and builds the input of the policy and value MLPs:
                           features = [grounding, observation, composed operator embedding]

    The other observation parts pass through unchanged and the composed operator embedding (size
    output_dim) is appended last: the same layout as the other embedding conditions, so they differ
    only in how the operator embedding is produced.

    The composed embedding is normalized to zero mean and magnitude (L2 norm) 1 (operator_embedding).
    Unnormalized, its magnitude is set by the architecture, not the operator: at initialization it is
    about 50-150 for the tree architecture and 20-30 for slots and geometric (stacked learned functions
    and DeepSets sums over literals amplify the components' scale), and training changes it further
    (the tree reached 170-570 over 1M PPO steps). The other inputs of the policy's MLP are around 1, so
    the embedding swamped them, and the unnormalized tree runs never learned pickup. Magnitude 1 also
    matches the random and text embeddings. `architecture` is TreeComposition, SlotComposition
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
        # No learnable gain or bias: a learnable gain could grow and make the magnitude large again.
        self.normalize = nn.LayerNorm(output_dim, elementwise_affine=False)
        self.unit_norm_scale = output_dim**-0.5  # LayerNorm output has norm sqrt(output_dim); this makes it 1

    def composed_embedding(self, structure: torch.Tensor) -> torch.Tensor:
        """(B, layout.size) structure arrays -> (B, output_dim) embeddings as the architecture composes them,
        before normalization. Their magnitude is free to grow during training (logged as
        operator_embedding_magnitude_before_normalization), but the policy never sees it."""
        return self.composition(unpack(structure.float(), self.layout))

    def operator_embedding(self, structure: torch.Tensor) -> torch.Tensor:
        """(B, layout.size) structure arrays -> (B, output_dim) composed embeddings with zero mean and
        magnitude (L2 norm) 1, up to LayerNorm's eps: what the policy sees."""
        return self.normalize(self.composed_embedding(structure)) * self.unit_norm_scale

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [observations[key].float().flatten(start_dim=1) for key in self.other_keys]
        # A batch holds many observations but only a few distinct operators (one per skill), so each distinct
        # structure is composed once and its embedding indexed back to every row: the same result and
        # gradients (they accumulate through the index) at a fraction of the cost.
        distinct, rows = torch.unique(observations[OPERATOR_EMBEDDING_KEY], dim=0, return_inverse=True)
        parts.append(self.operator_embedding(distinct)[rows])
        return torch.cat(parts, dim=1)


# Former name, kept so that models saved under it (model.zip stores the class by name) still load.
CompositionalOperatorExtractor = CompositionalPolicyFeaturesExtractor
