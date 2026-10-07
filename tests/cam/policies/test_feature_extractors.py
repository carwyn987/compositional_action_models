import json

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from main import main, parse_args, run_name  # noqa: E402
from cam.policies.feature_extractors import TrainableOperatorEmbeddingExtractor  # noqa: E402

SPACE = gym.spaces.Dict(
    {
        "observation": gym.spaces.Box(-np.inf, np.inf, (5,)),
        "operator_embedding": gym.spaces.Box(-np.inf, np.inf, (4,)),
        "grounding": gym.spaces.Box(-np.inf, np.inf, (3,)),
    }
)


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
