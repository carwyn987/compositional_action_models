"""Stable-Baselines3 features extractors that make the operator embedding trainable, where a policy keeps them, and
the SAC policy that chooses which loss trains one its actor and critic share."""

import gymnasium as gym
import torch
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import Actor, MultiInputPolicy
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


def features_extractors(policy: nn.Module) -> dict[str, nn.Module]:
    """A Stable-Baselines3 policy's features extractors by role, each distinct module once (a shared one is listed
    under its first role only):
        actor          SAC's actor; PPO's policy (pi)
        critic         SAC's critic; PPO's value function (vf)
        critic-target  SAC only: the target critic's copy, which follows the critic by Polyak averaging
    A SAC policy keeps its extractors in its actor and critics; its own features_extractor is None."""
    if hasattr(policy, "actor"):
        candidates = {"actor": policy.actor.features_extractor, "critic": policy.critic.features_extractor,
                      "critic-target": policy.critic_target.features_extractor}
    else:
        candidates = {"actor": policy.pi_features_extractor, "critic": policy.vf_features_extractor}
    extractors: dict[str, nn.Module] = {}
    for role, extractor in candidates.items():
        if all(extractor is not seen for seen in extractors.values()):
            extractors[role] = extractor
    return extractors


class DetachedFeaturesActor(Actor):
    """A SAC actor that reads its features without gradients, so the actor loss neither trains the features extractor
    nor runs its backward pass through it."""

    def extract_features(self, obs: dict[str, torch.Tensor], features_extractor: BaseFeaturesExtractor) -> torch.Tensor:
        with torch.no_grad():
            return super().extract_features(obs, features_extractor)


class FeaturesTrainingCritic(ContinuousCritic):
    """A SAC critic whose loss trains the features extractor it shares with the actor. SAC evaluates the critic on
    replayed actions for the critic loss, and on the actor's own actions, which carry the actor's gradient, for the
    actor loss (SAC.train, MultiTaskSAC.train): the features are computed with gradients for the first only, so the
    actor loss does not reach the extractor through the critic either."""

    def forward(self, obs: dict[str, torch.Tensor], actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not actions.requires_grad):
            features = self.extract_features(obs, self.features_extractor)
        return tuple(q_net(torch.cat([features, actions], dim=1)) for q_net in self.q_networks)


class OperatorEmbeddingSACPolicy(MultiInputPolicy):
    """Stable-Baselines3's SAC MultiInputPolicy, with operator_embedding_loss choosing the loss that trains a features
    extractor the actor and critic share (share_features_extractor=True, as main.py's policy_kwargs set for a learned
    operator embedding):
        critic  the critic's TD loss: the extractor is in the critic's optimizer, and the actor reads its features
                without gradients, in its own forward pass and through the critic's. The TD loss fits the value of
                every replayed transition, a richer signal than the actor loss, which follows only the critic's slope
                in the action; SAC from pixels (SAC-AE, DrQ) trains its shared encoder this way.
        actor   the actor loss, as Stable-Baselines3's own policy does: the extractor is in the actor's optimizer,
                and the critic reads its features without gradients.
    Either way the target critic has its own copy of the extractor, which follows the shared one by Polyak averaging.
    Extractors that are not shared are each trained by their own network's loss, whatever the option. The option is
    saved with the model (in its policy_kwargs), so a loaded or resumed model trains as it was created."""

    def __init__(self, *args, operator_embedding_loss: str = "critic", **kwargs):
        if operator_embedding_loss not in ("critic", "actor"):
            raise ValueError(f"operator_embedding_loss must be 'critic' or 'actor', not {operator_embedding_loss!r}")
        self.operator_embedding_loss = operator_embedding_loss
        super().__init__(*args, **kwargs)

    @property
    def critic_trains_features_extractor(self) -> bool:
        return self.share_features_extractor and self.operator_embedding_loss == "critic"

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        if self.critic_trains_features_extractor:
            # Stable-Baselines3's optimizers, with the shared extractor moved from the actor's to the critic's
            self.actor.optimizer = self.optimizer_class(
                [parameter for name, parameter in self.actor.named_parameters() if "features_extractor" not in name],
                lr=lr_schedule(1), **self.optimizer_kwargs,
            )
            self.critic.optimizer = self.optimizer_class(
                self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs,
            )

    def make_actor(self, features_extractor: BaseFeaturesExtractor | None = None) -> Actor:
        if not self.critic_trains_features_extractor:
            return super().make_actor(features_extractor)
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return DetachedFeaturesActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: BaseFeaturesExtractor | None = None) -> ContinuousCritic:
        if not self.critic_trains_features_extractor:
            return super().make_critic(features_extractor)
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return FeaturesTrainingCritic(**critic_kwargs).to(self.device)
