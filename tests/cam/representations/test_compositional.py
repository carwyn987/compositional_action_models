"""Compositional operator embeddings (representations/compositional/): the structure array, the three
architectures, and training through main."""

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO  # noqa: E402

from cam.domain.pddl.pddl_parser import parse_operator  # noqa: E402
from cam.representations.compositional.extractor import CompositionalOperatorExtractor  # noqa: E402
from cam.representations.compositional.geometric import GeometricComposition  # noqa: E402
from cam.representations.compositional.slots import SlotComposition  # noqa: E402
from cam.representations.compositional.structure import (  # noqa: E402
    OperatorLayout,
    StructuredOperatorEncoder,
    unpack,
)
from cam.representations.compositional.tree import TreeComposition  # noqa: E402
from cam.representations.text_backends import MockTextBackend  # noqa: E402
from main import main, parse_args, run_name  # noqa: E402
from tests.cam.domain.conftest import LEGACY_OPERATORS  # noqa: E402

ARITIES = {"holding": 1, "on-table": 1, "clear": 1, "on": 2, "gripper-empty": 0}  # ids: sorted names + 1
LAYOUT = OperatorLayout.from_predicate_arities(ARITIES, ["block"], max_predicate_arity=2, max_parameters=3, max_literals=8)
PICKUP, STACK = parse_operator(LEGACY_OPERATORS["pickup"]), parse_operator(LEGACY_OPERATORS["stack"])
ARCHITECTURES = [TreeComposition, SlotComposition, GeometricComposition]


def structure(*operators, layout=LAYOUT):
    return torch.as_tensor(np.stack([StructuredOperatorEncoder(layout).encode(op) for op in operators]))


def on_effect(first: str, second: str):
    """A two-block operator whose only effect is (on first second)."""
    return parse_operator(
        f"(:action op :parameters (?x - block ?y - block) :precondition (and) :effect (and (on {first} {second})))"
    )


@pytest.mark.unit
def test_structure_of_pickup():
    """Parameter types, then literal rows sorted by (kind, predicate, arguments), EMPTY (0) padding."""
    parts = unpack(structure(PICKUP), LAYOUT)
    assert parts.types.tolist() == [[1, 0, 0]]
    # clear=1, gripper-empty=2, holding=3, on-table=5; kinds: PRECONDITION 3, ADD 5, DELETE 6, EMPTY 0
    assert parts.kinds.tolist() == [[3, 3, 3, 5, 6, 6, 0, 0]]
    assert parts.predicates.tolist() == [[1, 2, 5, 3, 2, 5, 0, 0]]
    assert parts.arguments[0, :, 0].tolist() == [1, 0, 1, 1, 0, 1, 0, 0]  # ?o is parameter 1; 0 = no argument
    assert parts.name.shape == (1, 0)


@pytest.mark.unit
def test_structure_ignores_variable_names_and_literal_order():
    """Renaming variables or reordering literals in the PDDL source gives the same structure array."""
    renamed = parse_operator(LEGACY_OPERATORS["stack"].replace("?o", "?first").replace("?b", "?second"))
    reordered = dataclasses.replace(STACK, effects=STACK.effects[::-1])
    torch.testing.assert_close(structure(STACK), structure(renamed))
    torch.testing.assert_close(structure(STACK), structure(reordered))


@pytest.mark.unit
@pytest.mark.parametrize(
    "operator, layout",
    [
        ("(:action op :parameters (?x - block) :precondition (and) :effect (and (painted ?x)))", LAYOUT),
        ("(:action op :parameters (?x - block) :precondition (and) :effect (and (holding table)))", LAYOUT),
        ("(:action op :parameters (?x - cup) :precondition (and) :effect (and))", LAYOUT),
        (LEGACY_OPERATORS["stack"], dataclasses.replace(LAYOUT, max_parameters=1)),
        (LEGACY_OPERATORS["stack"], dataclasses.replace(LAYOUT, max_literals=2)),
    ],
    ids=["unknown-predicate", "constant-argument", "unknown-type", "too-many-parameters", "too-many-literals"],
)
def test_operators_the_layout_cannot_hold_are_rejected(operator, layout):
    with pytest.raises(ValueError):
        StructuredOperatorEncoder(layout).encode(parse_operator(operator))


@pytest.mark.unit
def test_name_text_embedding_is_appended():
    """With name_dim > 0 the array ends with the name's text embedding; the backend size must match."""
    layout = dataclasses.replace(LAYOUT, name_dim=16)
    array = StructuredOperatorEncoder(layout, MockTextBackend(16)).encode(PICKUP)
    np.testing.assert_array_equal(array[-16:], MockTextBackend(16).embed_text("pickup"))
    with pytest.raises(ValueError):
        StructuredOperatorEncoder(layout, MockTextBackend(8))


@pytest.mark.unit
@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("name_dim", [0, 16])
def test_architectures_embed_and_backpropagate_to_components(architecture, name_dim):
    """Each architecture maps a batch of structures to (B, output_dim), and gradients reach the
    component embeddings (here: predicates)."""
    layout = dataclasses.replace(LAYOUT, name_dim=name_dim)
    x = torch.as_tensor(np.stack([
        StructuredOperatorEncoder(layout, MockTextBackend(16) if name_dim else None).encode(op) for op in (PICKUP, STACK)
    ]))
    model = architecture(layout, 16, 32)
    embedding = model(unpack(x, layout))
    assert embedding.shape == (2, 32) and torch.isfinite(embedding).all()
    embedding.sum().backward()
    assert model.components.predicate_embedding.weight.grad.abs().sum() > 0


@pytest.mark.unit
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_argument_order_matters(architecture):
    """(on ?x ?y) and (on ?y ?x) get different embeddings: arguments are composed in order."""
    model = architecture(LAYOUT, 16, 32)
    embeddings = model(unpack(structure(on_effect("?x", "?y"), on_effect("?y", "?x")), LAYOUT))
    assert not torch.allclose(embeddings[0], embeddings[1])


@pytest.mark.unit
@pytest.mark.parametrize("architecture", [TreeComposition, SlotComposition])
def test_tree_and_slots_ignore_literal_order_and_padding(architecture):
    """Permuting literal rows, or holding the operator in a layout with more literal slots (same
    weights), leaves the embedding unchanged: empty slots are masked out."""
    torch.manual_seed(0)
    model = architecture(LAYOUT, 16, 32)
    x = structure(STACK)
    rows = x[:, LAYOUT.max_parameters :].reshape(1, LAYOUT.max_literals, LAYOUT.literal_width)
    permuted = torch.cat([x[:, : LAYOUT.max_parameters], rows[:, [4, 0, 3, 1, 2, 5, 6, 7]].flatten(1)], dim=1)
    torch.testing.assert_close(model(unpack(x, LAYOUT)), model(unpack(permuted, LAYOUT)))

    larger = dataclasses.replace(LAYOUT, max_literals=12)
    larger_model = architecture(larger, 16, 32)
    larger_model.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(unpack(x, LAYOUT)), larger_model(unpack(structure(STACK, layout=larger), larger)))


@pytest.mark.unit
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_extractor_embedding_has_unit_norm_even_when_the_composition_grows(architecture):
    """The extractor normalizes the composed embedding to zero mean and unit norm, so its scale cannot
    explode during training (the tree's grew ~170x in 2048 PPO steps before this): here the
    composition's output is scaled up 1000x and the embedding the policy sees is unchanged."""
    import gymnasium as gym

    space = gym.spaces.Dict({
        "observation": gym.spaces.Box(-np.inf, np.inf, (5,)),
        "operator_embedding": gym.spaces.Box(-np.inf, np.inf, (LAYOUT.size,)),
    })
    extractor = CompositionalOperatorExtractor(space, architecture, LAYOUT, 16, 32)
    x = structure(PICKUP, STACK)
    embedding = extractor.operator_embedding(x)
    # Norm 1 up to LayerNorm's eps (1e-5), which slightly shrinks outputs whose variance is small.
    torch.testing.assert_close(embedding.norm(dim=1), torch.ones(2), atol=1e-3, rtol=0)
    torch.testing.assert_close(embedding.mean(dim=1), torch.zeros(2), atol=1e-6, rtol=0)
    with torch.no_grad():
        for parameter in extractor.composition.parameters():
            parameter.mul_(1000.0 ** (1 / 8))  # inflate the composition's weights; output grows by orders of magnitude
    torch.testing.assert_close(extractor.operator_embedding(x).norm(dim=1), torch.ones(2), atol=1e-3, rtol=0)


@pytest.mark.unit
def test_layout_rejects_predicates_above_the_maximum_arity():
    """max_predicate_arity is a fixed maximum: an environment predicate with more arguments is an error."""
    with pytest.raises(ValueError):
        OperatorLayout.from_predicate_arities(ARITIES, ["block"], max_predicate_arity=1, max_parameters=3, max_literals=8)


@pytest.mark.unit
def test_compositional_embeddings_are_always_trainable():
    with pytest.raises(SystemExit):
        parse_args(["--skills", "pickup", "--operator-encoder", "compositional", "--trainable-operator-embedding"])


@pytest.mark.integration
@pytest.mark.parametrize("architecture", ["tree", "slots", "geometric"])
def test_short_training_run_with_each_architecture(tmp_path, architecture):
    """A short PPO run with --operator-encoder compositional trains, saves and reloads the policy, whose
    features extractor holds the chosen architecture."""
    argv = ["--skills", "pickup", "putdown", "--algorithm", "ppo", "--num-blocks", "2", "--max-steps-per-episode",
            "20", "--operator-encoder", "compositional", "--compositional-architecture", architecture,
            "--component-embedding-dim", "8", "--total-timesteps", "1024", "--mode", "train",
            "--output-dir", str(tmp_path)]
    main(argv)
    run = tmp_path / run_name(parse_args(argv))
    assert run.name.split("_")[2] == f"compositional-{architecture}"
    extractor = PPO.load(run / "model.zip").policy.features_extractor
    assert isinstance(extractor, CompositionalOperatorExtractor)
    expected = {"tree": TreeComposition, "slots": SlotComposition, "geometric": GeometricComposition}[architecture]
    assert isinstance(extractor.composition, expected)
