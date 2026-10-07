"""Stable-Baselines3 features extractors that make the operator embedding trainable."""

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

OPERATOR_EMBEDDING_KEY = "operator_embedding"


class TrainableOperatorEmbeddingExtractor(BaseFeaturesExtractor):
    """Makes only the operator embedding trainable: features = [grounding, observation, W @ operator_embedding].

    The other observation parts are passed through unchanged (flattened, in key order), and the operator
    embedding, multiplied by a learnable square matrix W (no bias, initialised to the identity), is
    appended last. This is the same layout as Stable-Baselines3's default extractor uses for the fixed
    embedding conditions, so the only difference between fixed and trainable conditions is W.

    Being part of the policy, W is trained by the RL losses. At initialisation the embedding is exactly
    the encoder's output, so training starts from it:
        random encoder    -> a learnable vector per operator, starting at its random vector
                             (independent per operator while #operators <= dim, as random vectors are
                             then linearly independent; updates couple them slightly through their overlap)
        multi-hot encoder -> learnable compositional embedding: W @ multi_hot is a sum of learned
                             vectors, one per literal feature present
    """

    def __init__(self, observation_space: gym.spaces.Dict):
        self.other_keys = sorted(key for key in observation_space.spaces if key != OPERATOR_EMBEDDING_KEY)
        embedding_dim = observation_space[OPERATOR_EMBEDDING_KEY].shape[0]
        other_dim = sum(int(observation_space[key].shape[0]) for key in self.other_keys)
        super().__init__(observation_space, features_dim=other_dim + embedding_dim)
        self.operator_embedding = nn.Linear(embedding_dim, embedding_dim, bias=False)
        with torch.no_grad():
            self.operator_embedding.weight.copy_(torch.eye(embedding_dim))

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [observations[key].float().flatten(start_dim=1) for key in self.other_keys]
        parts.append(self.operator_embedding(observations[OPERATOR_EMBEDDING_KEY].float()))
        return torch.cat(parts, dim=1)
