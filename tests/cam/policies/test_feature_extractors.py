import json

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO, SAC  # noqa: E402
from stable_baselines3.sac.policies import MultiInputPolicy  # noqa: E402

from main import main, parse_args, run_name  # noqa: E402
from cam.policies.feature_extractors import OperatorEmbeddingSACPolicy, TrainableOperatorEmbeddingExtractor  # noqa: E402
from cam.training.multi_task_sac import MultiTaskSAC  # noqa: E402
from tests.cam.training.test_multi_task_sac import TwoSkillBandit  # noqa: E402

SPACE = gym.spaces.Dict(
    {
        "observation": gym.spaces.Box(-np.inf, np.inf, (5,)),
        "operator_embedding": gym.spaces.Box(-np.inf, np.inf, (4,)),
        "grounding": gym.spaces.Box(-np.inf, np.inf, (3,)),
    }
)
# What one gradient step does to an operator embedding the actor and critic share (embedding_updates), when the
# critic loss trains it and when the actor loss does
CRITIC_LOSS_TRAINS_IT = ["gradient", "critic step changes W", "actor step"]
ACTOR_LOSS_TRAINS_IT = ["critic step", "gradient", "actor step changes W"]


@pytest.mark.unit
def test_starts_as_the_encoder_output():
    """W is initialised to the identity, so at the start the features are the plain concatenation:
    grounding, observation, then the operator embedding unchanged."""
    extractor = TrainableOperatorEmbeddingExtractor(SPACE)
    obs = {key: torch.randn(2, space.shape[0]) for key, space in SPACE.spaces.items()}
    features = extractor(obs)
    assert extractor.features_dim == 12 and features.shape == (2, 12)
    torch.testing.assert_close(features, torch.cat([obs["grounding"], obs["observation"], obs["operator_embedding"]], 1))


@pytest.mark.integration
def test_training_updates_the_embedding_and_the_model_reloads(tmp_path):
    """A short PPO run with --operator-encoder random --trainable-operator-embedding changes W away from
    the identity (gradients reach the embedding), and the saved model reloads with the trained W."""
    argv = ["--skills", "pickup", "putdown", "--algorithm", "ppo", "--num-blocks", "2", "--max-steps-per-episode",
            "20", "--operator-encoder", "random", "--trainable-operator-embedding", "--total-timesteps", "1024",
            "--mode", "train", "--output-dir", str(tmp_path)]
    main(argv)
    run = tmp_path / run_name(parse_args(argv))
    assert run.name.split("_")[2] == "random-trainable"
    assert json.loads((run / "config.json").read_text())["trainable_operator_embedding"] is True
    weight = PPO.load(run / "model.zip").policy.features_extractor.operator_embedding.weight.detach()
    assert not torch.allclose(weight, torch.eye(weight.shape[0]))


@pytest.mark.integration
def test_a_sac_run_saves_the_operator_embedding_loss_it_is_given(tmp_path):
    """A short SAC run with --operator-embedding-loss actor saves it in config.json and in its policy. (The default,
    critic, is also the trainer's own, so it is the other value that shows main.py passes the option on.)"""
    argv = ["--skills", "pickup", "--algorithm", "sac", "--num-blocks", "2", "--max-steps-per-episode", "10",
            "--operator-encoder", "random", "--trainable-operator-embedding", "--operator-embedding-loss", "actor",
            "--total-timesteps", "1100", "--mode", "train", "--output-dir", str(tmp_path)]
    main(argv)
    run = tmp_path / run_name(parse_args(argv))
    assert json.loads((run / "config.json").read_text())["operator_embedding_loss"] == "actor"
    policy = SAC.load(run / "model.zip", device="cpu").policy
    assert type(policy) is OperatorEmbeddingSACPolicy and policy.operator_embedding_loss == "actor"


def shared_embedding_sac(algorithm=SAC, policy=OperatorEmbeddingSACPolicy, **policy_options):
    """A SAC model on TwoSkillBandit whose actor and critic share a trainable operator embedding, as main.py builds one;
    it makes no gradient steps while it learns."""
    policy_kwargs = {"net_arch": [16, 16], "features_extractor_class": TrainableOperatorEmbeddingExtractor,
                     "share_features_extractor": True} | policy_options
    return algorithm(policy, TwoSkillBandit(), learning_starts=10_000, batch_size=32, seed=0, verbose=0, device="cpu",
                     policy_kwargs=policy_kwargs)


def embedding_updates(sac) -> list[str]:
    """What one gradient step, after 200 steps of collection, does to the shared operator embedding's W, in order: each
    gradient that reaches W, and each optimizer step, with whether it changes W."""
    sac.learn(200, reset_num_timesteps=False)
    weight = sac.actor.features_extractor.operator_embedding.weight
    events, before = [], weight.detach().clone()
    weight.register_hook(lambda gradient: events.append("gradient"))

    def record(step):
        def hook(optimizer, args, kwargs):
            nonlocal before
            events.append(step + ("" if torch.equal(weight, before) else " changes W"))
            before = weight.detach().clone()
        return hook

    sac.critic.optimizer.register_step_post_hook(record("critic step"))
    sac.actor.optimizer.register_step_post_hook(record("actor step"))
    sac.train(gradient_steps=1, batch_size=32)
    return events


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", [SAC, MultiTaskSAC])
@pytest.mark.parametrize("loss, expected", [("critic", CRITIC_LOSS_TRAINS_IT), ("actor", ACTOR_LOSS_TRAINS_IT)])
def test_one_loss_trains_the_operator_embedding_actor_and_critic_share(algorithm, loss, expected):
    """One gradient step of SAC and multi-task SAC, watched on the operator embedding's W. With operator_embedding_loss
    critic, the critic loss's gradient is the only one to reach W and the critic's step changes it, as that loss alone
    would; the actor loss's backward pass does not reach W, and the actor's step leaves it. Control, actor (what
    Stable-Baselines3's own policy does): the critic's step leaves W, and the actor loss's gradient and step change it.
    Either way the target critic's own copy of W follows it by Polyak averaging."""
    sac = shared_embedding_sac(algorithm, operator_embedding_loss=loss)
    start = sac.actor.features_extractor.operator_embedding.weight.detach().clone()
    assert embedding_updates(sac) == expected
    weight, target = (critic.features_extractor.operator_embedding.weight for critic in (sac.critic, sac.critic_target))
    assert weight is sac.actor.features_extractor.operator_embedding.weight and target is not weight
    torch.testing.assert_close(target, (1 - sac.tau) * start + sac.tau * weight)


@pytest.mark.unit
@pytest.mark.parametrize("policy, options, expected", [
    (OperatorEmbeddingSACPolicy, {"operator_embedding_loss": "critic"}, CRITIC_LOSS_TRAINS_IT),
    (OperatorEmbeddingSACPolicy, {"operator_embedding_loss": "actor"}, ACTOR_LOSS_TRAINS_IT),
    (MultiInputPolicy, {}, ACTOR_LOSS_TRAINS_IT),
])
def test_a_loaded_model_keeps_the_loss_that_trains_its_operator_embedding(tmp_path, policy, options, expected):
    """The option is saved with the model, so loaded (as the scripts and a resumed run load it) a model trains its
    operator embedding with the same loss. A model of Stable-Baselines3's own MultiInputPolicy, as runs saved before
    the option have, loads as that and trains it with the actor loss."""
    shared_embedding_sac(MultiTaskSAC, policy, **options).save(tmp_path / "model.zip")
    loaded = MultiTaskSAC.load(tmp_path / "model.zip", env=TwoSkillBandit(), device="cpu")
    assert type(loaded.policy) is policy
    assert embedding_updates(loaded) == expected
