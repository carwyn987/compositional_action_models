"""Compositional operator embeddings (representations/compositional/): the structure array, the three
architectures, and training through main."""

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO  # noqa: E402

from cam.domain.pddl.pddl_parser import parse_operator  # noqa: E402
from cam.representations.compositional.extractor import CompositionalPolicyFeaturesExtractor  # noqa: E402
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
    """The extractor normalizes the composed embedding to zero mean and unit norm, so the composition's
    scale (about 100 at initialization for the tree, and changing during training) never reaches the
    policy: here the composition's output is scaled up 1000x and the embedding the policy sees is unchanged."""
    import gymnasium as gym

    space = gym.spaces.Dict({
        "observation": gym.spaces.Box(-np.inf, np.inf, (5,)),
        "operator_embedding": gym.spaces.Box(-np.inf, np.inf, (LAYOUT.size,)),
    })
    extractor = CompositionalPolicyFeaturesExtractor(space, architecture, LAYOUT, 16, 32)
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
    assert isinstance(extractor, CompositionalPolicyFeaturesExtractor)
    expected = {"tree": TreeComposition, "slots": SlotComposition, "geometric": GeometricComposition}[architecture]
    assert isinstance(extractor.composition, expected)


@pytest.mark.unit
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_extractor_composes_each_distinct_operator_once_with_the_same_result(architecture):
    """forward composes the distinct operators of a batch once and indexes them back to every row: the
    features and the gradients equal composing every row."""
    import gymnasium as gym

    space = gym.spaces.Dict({
        "observation": gym.spaces.Box(-np.inf, np.inf, (5,)),
        "operator_embedding": gym.spaces.Box(-np.inf, np.inf, (LAYOUT.size,)),
    })
    extractor = CompositionalPolicyFeaturesExtractor(space, architecture, LAYOUT, 16, 32)
    structures = structure(PICKUP, STACK)[[0, 1, 1, 0, 1]]  # 5 rows, 2 distinct operators
    observations = {"observation": torch.randn(5, 5), "operator_embedding": structures}
    features = extractor(observations)
    every_row = torch.cat([observations["observation"], extractor.operator_embedding(structures)], dim=1)
    torch.testing.assert_close(features, every_row)
    weights = torch.randn_like(features)
    gradients = torch.autograd.grad((features * weights).sum(), list(extractor.parameters()), allow_unused=True)
    expected = torch.autograd.grad((every_row * weights).sum(), list(extractor.parameters()), allow_unused=True)
    for gradient, reference in zip(gradients, expected):
        if reference is not None:
            torch.testing.assert_close(gradient, reference)


@pytest.mark.unit
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_intermediates_end_in_the_embedding(architecture):
    """forward returns intermediates' "operator" stage, with the name and without; for the tree, each stage is the
    function of the stage below it (AND = rho(and-sum), PRE(and pre), EFF(and eff)), and and-sum is the sum of the
    section's conjuncts."""
    layout = dataclasses.replace(LAYOUT, name_dim=16)
    x = torch.as_tensor(np.stack([StructuredOperatorEncoder(layout, MockTextBackend(16)).encode(op) for op in (PICKUP, STACK)]))
    model = architecture(layout, 16, 32)
    with torch.no_grad():
        stages = model.intermediates(unpack(x, layout))
        assert torch.equal(stages["operator"], model(unpack(x, layout)))
        if architecture is TreeComposition:
            parts = unpack(x, layout)
            effect = ((parts.kinds == 5) | (parts.kinds == 6))[..., None]  # ADD, DELETE
            torch.testing.assert_close(stages["and-sum eff"], (stages["conjunct"] * effect).sum(dim=1))
            torch.testing.assert_close(stages["and eff"], model.conjunction(stages["and-sum eff"]))
            torch.testing.assert_close(stages["pre"], model.precondition(stages["and pre"]))
            torch.testing.assert_close(stages["eff"], model.effect(stages["and eff"]))


@pytest.mark.unit
@pytest.mark.parametrize("stage_normalization", [False, True])
def test_one_literal_outgrowing_the_others_collapses_operators_only_without_stage_normalization(stage_normalization):
    """The collapse seen in training (docs/results.md, RQ3 experiment 2): pickup-raised and unstack-raised became one
    embedding, pickup and unstack nearly one, and only pickup and pickup-raised stayed apart. Growing one component,
    (raised ?o)'s predicate embedding, 1e4 times reproduces exactly that without stage normalization (the control):
    every operator with the literal, as (raised ?o) or (not (raised ?o)), becomes the point its sign gives. With it,
    each literal's conjunct keeps its magnitude and the embeddings do not move."""
    import gymnasium as gym

    from cam.experiments.environment_setup import OPERATOR_ENCODERS, compositional_layout
    from cam.skills.registry import build_skill

    config = parse_args(["--skills", "pickup", "--operator-encoder", "compositional", "--compositional-name", "none"])
    layout = compositional_layout(config)
    skills = [build_skill(name, config) for name in ("pickup", "pickup-raised", "unstack", "unstack-raised")]
    encoder = OPERATOR_ENCODERS["compositional"](config, skills)
    x = torch.as_tensor(np.stack([encoder.encode(skill.symbolic_action_model) for skill in skills]))
    space = gym.spaces.Dict({"observation": gym.spaces.Box(-np.inf, np.inf, (5,)),
                             "operator_embedding": gym.spaces.Box(-np.inf, np.inf, (layout.size,))})
    torch.manual_seed(0)
    extractor = CompositionalPolicyFeaturesExtractor(space, TreeComposition, layout, 32, 256,
                                                     {"stage_normalization": stage_normalization})

    def cosines() -> np.ndarray:
        """pickup-raised / unstack-raised, pickup / unstack, pickup / pickup-raised."""
        with torch.no_grad():
            embeddings = torch.nn.functional.normalize(extractor.operator_embedding(x), dim=1)
        similarity = (embeddings @ embeddings.T).numpy()
        return np.array([similarity[1, 3], similarity[0, 2], similarity[0, 1]])

    before = cosines()
    with torch.no_grad():
        extractor.composition.components.predicate_embedding.weight[layout.predicates.index("raised") + 1] *= 1e4
    after = cosines()
    if stage_normalization:
        np.testing.assert_allclose(after, before, atol=0.01)
    else:
        assert after[0] > 0.9999 and after[1] > 0.999 and after[2] < 0.9 < before[2]


@pytest.mark.unit
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_stage_normalization_adds_no_parameters(architecture):
    """Its LayerNorms have no gain or bias, so a model saved without it has exactly the parameters, and without
    the option (the default) computes exactly as it was trained."""
    def parameters(**options):
        return {name: tuple(p.shape) for name, p in architecture(LAYOUT, 16, 32, **options).named_parameters()}

    assert parameters(stage_normalization=True) == parameters() == parameters(stage_normalization=False)
