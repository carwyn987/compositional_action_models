"""SAC with a separate entropy temperature per skill: multi-task SAC (MT-SAC, as used on Meta-World).

SAC tunes its temperature (alpha, the entropy coefficient) so that the policy's entropy stays near a target.
With one temperature for every skill, the target holds only on average over the skills: skills the policy has
learned (narrow, low entropy) and skills it is still exploring (broad) pull the one temperature in opposite
directions, so it suits neither, and a hard skill can stop exploring because the easy ones dominate.
MultiTaskSAC keeps one temperature per skill, each tuned on that skill's transitions only.

A transition's skill is its operator embedding (obs["operator_embedding"]): each distinct embedding gets its own
temperature slot the first time a batch contains it. The policy's input is unchanged; only the training
objective uses which skill a transition belongs to. Operators with identical embeddings (e.g. under
--operator-identity-aliases) share a slot, as they share everything else.
"""

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F

from cam.policies.feature_extractors import OPERATOR_EMBEDDING_KEY


def embedding_key(embedding) -> str:
    """An operator embedding's exact float32 bytes, as hex (JSON-serializable, so saved with the model)."""
    return np.asarray(embedding, dtype=np.float32).tobytes().hex()


class MultiTaskSAC(SAC):
    """SAC with one learned temperature per operator embedding (up to max_skills of them). The temperatures are
    saved with the model (log_ent_coef, one entry per slot; skill_slots maps each embedding to its slot)."""

    def __init__(self, *args, max_skills: int = 64, **kwargs):
        self.max_skills = max_skills
        self.skill_slots: dict[str, int] = {}  # embedding_key(operator embedding) -> temperature slot
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        if self.ent_coef_optimizer is None:
            raise ValueError("MultiTaskSAC learns its temperatures: ent_coef must be 'auto' or 'auto_<initial value>'")
        initial = float(self.log_ent_coef.detach()[0])
        self.log_ent_coef = th.full((self.max_skills,), initial, device=self.device).requires_grad_(True)
        self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))

    def skill_slot_indices(self, operator_embeddings: th.Tensor) -> th.Tensor:
        """The temperature slot of each row's operator embedding, assigning new slots to new embeddings."""
        unique, inverse = th.unique(operator_embeddings, dim=0, return_inverse=True)
        slots = []
        for row in unique.cpu().numpy():
            key = embedding_key(row)
            if key not in self.skill_slots:
                if len(self.skill_slots) == self.max_skills:
                    raise RuntimeError(f"more than max_skills={self.max_skills} distinct operator embeddings")
                self.skill_slots[key] = len(self.skill_slots)
            slots.append(self.skill_slots[key])
        return th.as_tensor(slots, device=operator_embeddings.device)[inverse]

    def skill_temperatures(self) -> dict[str, float]:
        """Each named skill's temperature, named from the training environment's operator embeddings (skills
        sharing an embedding are joined with +); empty without an environment that has them."""
        try:
            embeddings = self.env.get_attr("operator_embeddings")[0] if self.env is not None else {}
        except AttributeError:
            embeddings = {}
        names: dict[int, list[str]] = {}
        for model, embedding in embeddings.items():
            slot = self.skill_slots.get(embedding_key(embedding))
            if slot is not None:
                names.setdefault(slot, []).append(model.name)
        temperatures = th.exp(self.log_ent_coef.detach()).cpu().numpy()
        return {"+".join(sorted(skills)): float(temperatures[slot]) for slot, skills in names.items()}

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """Stable-Baselines3's SAC.train (2.9), with each transition's entropy coefficient taken from its skill's
        temperature, and each temperature's loss over its skill's transitions."""
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # each transition's skill's temperature; the loss tunes each temperature on its skill's transitions
            slots = self.skill_slot_indices(replay_data.observations[OPERATOR_EMBEDDING_KEY])
            log_ent_coef = self.log_ent_coef[slots].reshape(-1, 1)
            ent_coef = th.exp(log_ent_coef.detach())
            ent_coef_loss = -(log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            ent_coef_losses.append(ent_coef_loss.item())
            ent_coefs.append(ent_coef.mean().item())

            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        for skill, temperature in self.skill_temperatures().items():
            self.logger.record(f"train/ent_coef/{skill}", temperature)
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
