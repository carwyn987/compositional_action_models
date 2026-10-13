import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

import torch  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

from cam.training.multi_task_sac import MultiTaskSAC, embedding_key  # noqa: E402

PRECISE, FREE = np.eye(2, dtype=np.float32)


class Operator:
    def __init__(self, name: str):
        self.name = name


class TwoSkillBandit(gym.Env):
    """One-step episodes alternating two skills, told apart only by their operator embedding: "precise" rewards
    actions near 0.5 sharply, "free" rewards nothing. Like PolicyObservationWrapper, it exposes its
    operator_embeddings (operator -> embedding), from which MultiTaskSAC names its temperatures."""

    observation_space = gym.spaces.Dict({
        "observation": gym.spaces.Box(-1, 1, (1,)), "operator_embedding": gym.spaces.Box(0, 1, (2,)),
    })
    action_space = gym.spaces.Box(-1, 1, (1,))

    def __init__(self):
        self.operator_embeddings = {Operator("precise"): PRECISE, Operator("free"): FREE}
        self.episodes = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.embedding = (PRECISE, FREE)[self.episodes % 2]
        self.episodes += 1
        return {"observation": np.zeros(1, np.float32), "operator_embedding": self.embedding}, {}

    def step(self, action):
        reward = -20.0 * float((action[0] - 0.5) ** 2) if self.embedding is PRECISE else 0.0
        return {"observation": np.zeros(1, np.float32), "operator_embedding": self.embedding}, reward, True, False, {}


def model(algorithm=MultiTaskSAC, **overrides):
    options = dict(learning_starts=100, batch_size=32, seed=0, verbose=0, device="cpu",
                   policy_kwargs={"net_arch": [16, 16]}) | overrides
    return algorithm("MultiInputPolicy", TwoSkillBandit(), **options)


def temperature(sac, embedding) -> float:
    """The temperature a skill trains with: its own slot in MultiTaskSAC, the one shared value in SAC."""
    if isinstance(sac, MultiTaskSAC):
        return float(torch.exp(sac.log_ent_coef.detach()[sac.skill_slots[embedding_key(embedding)]]))
    return float(torch.exp(sac.log_ent_coef.detach()[0]))


@pytest.mark.unit
def test_skills_with_the_same_embedding_share_a_slot():
    """Slots follow the operator embedding: identical embeddings (e.g. an aliased, repair-hidden operator) share one,
    a different embedding gets the next."""
    sac = model()
    slots = sac.skill_slot_indices(torch.as_tensor(np.stack([FREE, FREE, PRECISE, FREE])))
    assert slots.tolist() == [0, 0, 1, 0]


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", [SAC, MultiTaskSAC])
def test_a_skills_temperature_is_tuned_only_on_its_own_transitions(algorithm):
    """Train on transitions of the precise skill only. MultiTaskSAC leaves the free skill's temperature exactly where
    it was; SAC (the control) moves the one temperature every skill shares."""
    sac = model(algorithm, learning_starts=10_000)
    sac.learn(200)  # fills the replay buffer, no gradient steps yet
    buffer = sac.replay_buffer
    for observations in (buffer.observations, buffer.next_observations):
        observations["operator_embedding"][:] = PRECISE  # every stored transition is now the precise skill's
    if algorithm is MultiTaskSAC:
        sac.skill_slot_indices(torch.as_tensor(np.stack([PRECISE, FREE])))  # both skills have slots
    precise, free = temperature(sac, PRECISE), temperature(sac, FREE)
    sac.train(gradient_steps=20, batch_size=32)
    assert temperature(sac, PRECISE) != precise
    assert (temperature(sac, FREE) == free) == (algorithm is MultiTaskSAC)


@pytest.mark.integration
def test_each_skill_gets_the_temperature_its_own_entropy_needs():
    """After 1500 steps, the precise skill (its policy narrows, so its entropy falls toward the target) keeps a higher
    temperature than the free skill (its policy stays broad, above the target, so its temperature keeps falling).
    The control, SAC, gives both skills the same temperature."""
    shared, separate = model(SAC), model()
    for sac in (shared, separate):
        sac.learn(1500)
    assert temperature(shared, PRECISE) == temperature(shared, FREE)
    assert temperature(separate, PRECISE) > 1.05 * temperature(separate, FREE)
    assert set(separate.skill_temperatures()) == {"precise", "free"}


@pytest.mark.unit
def test_temperatures_and_slots_survive_save_and_load(tmp_path):
    """The per-skill temperatures and the embedding -> slot map are saved with the model."""
    sac = model(learning_starts=50)
    sac.learn(150)
    sac.save(tmp_path / "model.zip")
    loaded = MultiTaskSAC.load(tmp_path / "model.zip", device="cpu")
    assert loaded.skill_slots == sac.skill_slots
    assert torch.equal(loaded.log_ent_coef.detach(), sac.log_ent_coef.detach())
