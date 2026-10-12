"""Operator embedding magnitude (L2 norm, i.e. length) logging: the helper and the TensorBoard tags."""

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

import torch  # noqa: E402
from stable_baselines3.common.torch_layers import CombinedExtractor  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

from cam.policies.feature_extractors import TrainableOperatorEmbeddingExtractor  # noqa: E402
from cam.representations.compositional.extractor import CompositionalPolicyFeaturesExtractor  # noqa: E402
from cam.representations.compositional.structure import OperatorLayout  # noqa: E402
from cam.representations.compositional.tree import TreeComposition  # noqa: E402
from cam.training.stable_baselines3_trainer import operator_embedding_magnitudes  # noqa: E402
from main import main, parse_args, run_name  # noqa: E402

LAYOUT = OperatorLayout.from_predicate_arities({"holding": 1, "on": 2}, ["block"], 2, 2, 4)


def space(embedding_dim: int) -> gym.spaces.Dict:
    return gym.spaces.Dict({
        "observation": gym.spaces.Box(-np.inf, np.inf, (5,)),
        "operator_embedding": gym.spaces.Box(-np.inf, np.inf, (embedding_dim,)),
    })


@pytest.mark.unit
def test_fixed_embedding_extractor_has_no_magnitude_to_log():
    assert operator_embedding_magnitudes(CombinedExtractor(space(4)), np.ones((2, 4))) == {}


@pytest.mark.unit
def test_trainable_embedding_magnitude_is_the_norm_of_w_times_the_input():
    """W starts at the identity, so the magnitude starts as the input's L2 norm."""
    inputs = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
    magnitudes = operator_embedding_magnitudes(TrainableOperatorEmbeddingExtractor(space(3)), inputs)
    assert list(magnitudes) == ["operator_embedding_magnitude"]
    np.testing.assert_allclose(magnitudes["operator_embedding_magnitude"], [5.0, 2.0], rtol=1e-6)


@pytest.mark.unit
def test_compositional_logs_the_normalized_and_the_raw_magnitude():
    """What the policy sees has magnitude 1; the composition's raw output is reported separately."""
    extractor = CompositionalPolicyFeaturesExtractor(space(LAYOUT.size), TreeComposition, LAYOUT, 8, 16)
    structure = np.zeros((1, LAYOUT.size))
    structure[0, 0] = 1  # one block parameter
    structure[0, LAYOUT.max_parameters : LAYOUT.max_parameters + 3] = [3, 1, 1]  # precondition (holding ?o)
    with torch.no_grad():
        for parameter in extractor.composition.parameters():
            parameter.mul_(3.0)  # make the raw output clearly larger than 1
    magnitudes = operator_embedding_magnitudes(extractor, structure)
    np.testing.assert_allclose(magnitudes["operator_embedding_magnitude"], [1.0], atol=1e-3)
    assert magnitudes["operator_embedding_magnitude_before_normalization"][0] > 2.0


def logged_tags(run) -> set[str]:
    (events,) = (run / "tensorboard").glob("*")
    accumulator = EventAccumulator(str(events))
    accumulator.Reload()
    return set(accumulator.Tags()["scalars"])


@pytest.mark.integration
@pytest.mark.parametrize(
    "options, expected",
    [
        (["--algorithm", "ppo", "--operator-encoder", "compositional", "--compositional-architecture", "tree"],
         {"operator_embedding_magnitude/pickup", "operator_embedding_magnitude_before_normalization/pickup"}),
        (["--algorithm", "sac", "--operator-encoder", "random", "--trainable-operator-embedding"],
         {"operator_embedding_magnitude/actor/pickup", "operator_embedding_magnitude/critic/putdown"}),
    ],
)
def test_training_logs_operator_embedding_magnitude_to_tensorboard(tmp_path, options, expected):
    arguments = ["--skills", "pickup", "putdown", *options, "--num-blocks", "2", "--max-steps-per-episode", "10",
                 "--total-timesteps", "1100", "--mode", "train", "--output-dir", str(tmp_path)]
    main(arguments)
    assert expected <= logged_tags(tmp_path / run_name(parse_args(arguments)))
