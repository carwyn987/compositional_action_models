"""embedding_points: each source's points against answers known from the operators, with the controls that show what
each source does not encode."""

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

import torch  # noqa: E402

from cam.domain.pddl.pddl_parser import parse_operator  # noqa: E402
from cam.domain.pddl.pddl_writer import to_pddl  # noqa: E402
from cam.experiments.embedding_points import (  # noqa: E402
    CompositionalSource,
    FixedEncoderSource,
    LabelledOperator,
    TrainableRunSource,
    make_offline,
    parse_expression,
    registered_operators,
)
from cam.experiments.environment_setup import compositional_layout  # noqa: E402
from cam.policies.feature_extractors import OPERATOR_EMBEDDING_KEY  # noqa: E402
from cam.representations.text_backends import OpenAIBackend  # noqa: E402
from main import parse_args, policy_kwargs  # noqa: E402

REGISTERED = registered_operators()
BY_NAME = {entry.label: entry.operator for entry in REGISTERED}


def config(*argv: str) -> dict:
    return parse_args(["--skills", "pickup", "--compositional-name", "none", *argv])


def vectors(source, level: str, operators=REGISTERED) -> dict[str, np.ndarray]:
    points, _ = source.points(level, list(operators))
    return {point.label: point.vector.astype(np.float64) for point in points}


def compositional(architecture: str = "tree", seed: int = 0, *argv: str) -> CompositionalSource:
    settings = config("--operator-encoder", "compositional", "--compositional-architecture", architecture, *argv)
    kwargs = policy_kwargs(settings)
    space = gym.spaces.Dict({"observation": gym.spaces.Box(-1, 1, (1,)), OPERATOR_EMBEDDING_KEY: gym.spaces.Box(
        -np.inf, np.inf, (compositional_layout(settings).size,))})
    torch.manual_seed(seed)
    extractor = kwargs["features_extractor_class"](space, **kwargs["features_extractor_kwargs"])
    return CompositionalSource(f"compositional-{architecture}", settings, extractor)


def custom(label: str, pddl: str) -> LabelledOperator:
    return LabelledOperator(label, parse_operator(pddl), "custom")


def cosine(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.unit
def test_multi_hot_distances_are_the_shared_features():
    """pickup and pickup-raised differ in one literal (delete raised -> add raised): 14 of their 16 features shared,
    cosine 14/16 and distance 2; pickup and unstack (18 features) share 12: cosine 12/sqrt(16 x 18), distance
    sqrt(10). The control, one-hot, has cosine 0 for both: it encodes no structure."""
    multi_hot = vectors(FixedEncoderSource("multi-hot", config("--operator-encoder", "multi-hot")), "operator")
    assert cosine(multi_hot["pickup"], multi_hot["pickup-raised"]) == pytest.approx(14 / 16)
    assert np.linalg.norm(multi_hot["pickup"] - multi_hot["pickup-raised"]) == pytest.approx(2)
    assert cosine(multi_hot["pickup"], multi_hot["unstack"]) == pytest.approx(12 / np.sqrt(288))
    assert np.linalg.norm(multi_hot["pickup"] - multi_hot["unstack"]) == pytest.approx(np.sqrt(10))
    one_hot = vectors(FixedEncoderSource("one-hot", config("--operator-encoder", "one-hot")), "operator")
    assert cosine(one_hot["pickup"], one_hot["pickup-raised"]) == cosine(one_hot["pickup"], one_hot["unstack"]) == 0


@pytest.mark.unit
def test_a_custom_copy_of_an_operator_lands_on_it_and_a_changed_one_moves_where_content_counts():
    """A custom point with pickup's PDDL lands exactly on pickup in every source. Written with its literals reordered
    and its variable renamed, it still does in every source that sees structure or names only (multi-hot, the
    compositional tree, one-hot, random), but not in text, which embeds the PDDL as written. Changing one literal
    moves it in the content-based sources and not in one-hot or random (the control: they see only the name);
    renaming the operator moves random, and the tree only with the name included."""
    pickup = to_pddl(BY_NAME["pickup"])
    rewritten = ("(:action pickup :parameters (?b - block) :precondition (and (gripper-empty) (clear ?b) (on-table ?b)) "
                 ":effect (and (not (gripper-empty)) (holding ?b) (not (clear ?b)) (not (on-table ?b)) (not (raised ?b))))")
    points = [custom("copy", pickup), custom("rewritten", rewritten),
              custom("changed", pickup.replace("(not (raised ?o))", "(raised ?o)")),
              custom("renamed", pickup.replace("pickup", "lift"))]
    sources = {name: FixedEncoderSource(name, config("--operator-encoder", name)) for name in ("multi-hot", "text", "one-hot", "random")}
    sources["tree"], sources["tree with name"] = compositional(), compositional("tree", 0, "--compositional-name", "text")
    moved = {}
    for name, source in sources.items():
        found = vectors(source, "operator", REGISTERED + points)
        distance = {label: float(np.linalg.norm(found[label] - found["pickup"])) if label in found else None
                    for label in ("copy", "rewritten", "changed", "renamed")}
        assert distance["copy"] == 0, name
        moved[name] = {label: value is None or value > 1e-6 for label, value in distance.items()}
    assert {name: moved[name]["rewritten"] for name in sources} == {
        "multi-hot": False, "text": True, "one-hot": False, "random": False, "tree": False, "tree with name": False}
    assert {name: moved[name]["changed"] for name in sources} == {
        "multi-hot": True, "text": True, "one-hot": False, "random": False, "tree": True, "tree with name": True}
    assert {name: moved[name]["renamed"] for name in ("multi-hot", "random", "tree", "tree with name")} == {
        "multi-hot": False, "random": True, "tree": False, "tree with name": True}
    _, unencodable = sources["one-hot"].points("operator", points)
    assert list(unencodable) == ["renamed"]  # one-hot knows only the registered names


@pytest.mark.unit
def test_multi_hot_through_w_is_a_sum_of_columns():
    """With multi-hot and a learned W (here random), each operator is the sum of W's columns for its features, and of
    its literals' points. The control: push-beside has two preconditions on one predicate (on-table ?o, on-table ?u),
    so its literals count that predicate's name-only feature twice and the operator once."""
    settings = config("--operator-encoder", "multi-hot", "--trainable-operator-embedding", "--algorithm", "sac")
    extractor = type("Extractor", (), {"operator_embedding": torch.nn.Linear(256, 256, bias=False)})()
    torch.nn.init.normal_(extractor.operator_embedding.weight)
    source = TrainableRunSource("trained", settings, extractor, ["pickup"])
    operator = vectors(source, "operator")
    features, literals = source.points("feature", REGISTERED)[0], source.points("literal", REGISTERED)[0]

    def total(points, label):
        return sum(point.vector.astype(np.float64) for point in points if label in point.operators)

    for label in operator:
        np.testing.assert_allclose(total(features, label), operator[label], rtol=1e-5, atol=1e-4)
        assert np.allclose(total(literals, label), operator[label], rtol=1e-5, atol=1e-4) == (label != "push-beside")
    assert set(vectors(source, "input")) == set(BY_NAME)  # x, the control a trainable run is compared with


@pytest.mark.unit
def test_the_trees_and_sum_is_linear_in_literals_and_its_operator_is_not():
    """The tree's and-sum is a sum over literals, so pickup -> pickup-raised and unstack -> unstack-raised (each: delete
    raised -> add raised) move the effect's and-sum by the same vector, phi(raised) - phi(NOT raised): analogy cosine
    1. The control: the composed operator embedding, after the nonlinear functions, does not move in parallel."""
    tree = compositional()
    sums = vectors(tree, "and-sum")
    conjuncts = vectors(tree, "conjunct")
    first = sums["pickup-raised eff"] - sums["pickup eff"]
    second = sums["unstack-raised eff"] - sums["unstack eff"]
    np.testing.assert_allclose(first, conjuncts["(raised ?o)"] - conjuncts["(not (raised ?o))"], atol=1e-5)
    assert cosine(first, second) == pytest.approx(1, abs=1e-6)
    operator = vectors(tree, "operator")
    assert cosine(operator["pickup-raised"] - operator["pickup"], operator["unstack-raised"] - operator["unstack"]) < 0.999


@pytest.mark.unit
@pytest.mark.parametrize("architecture", ["tree", "slots", "geometric"])
def test_every_compositional_level_is_computed_and_spaces_agree_in_size(architecture):
    """Every level has points; levels sharing a space have the same size; the operator level is exactly what the
    extractor gives the policy."""
    source = compositional(architecture)
    sizes = {}
    for level in source.levels():
        points, unencodable = source.points(level.name, REGISTERED)
        assert points and not unencodable, level.name
        assert {p.vector.shape for p in points} == {(level.dim,)}, level.name
        assert sizes.setdefault(level.space, level.dim) == level.dim, level.name
    structure = torch.as_tensor(np.stack([source.encoder.encode(entry.operator) for entry in REGISTERED]))
    with torch.no_grad():
        expected = source.extractor.operator_embedding(structure).numpy()
    found = vectors(source, "operator")
    np.testing.assert_array_equal(np.stack([found[entry.label] for entry in REGISTERED]), expected)


@pytest.mark.unit
def test_atoms_are_shared_across_sections_and_all_atoms_adds_the_rest():
    """An atom is one point whatever its section and sign: (clear ?o) is a precondition of pickup and a delete effect
    too. --all-atoms adds every predicate on every ordering of parameter positions (here 37 for 11 predicates over 3
    positions), keeping the operators' names for the atoms they have."""
    tree = compositional()
    atoms = tree.points("atom", REGISTERED)[0]
    assert sum(point.label == "(clear ?o)" for point in atoms) == 1
    tree.all_atoms = True
    every = tree.points("atom", REGISTERED)[0]
    assert len(every) == 37 and {point.label for point in atoms} <= {point.label for point in every}


@pytest.mark.unit
def test_offline_text_serves_the_cache_and_never_calls_the_api(tmp_path, monkeypatch):
    """--offline: a cached text is served, an uncached one is reported unencodable, and the API is never called."""
    def no_api(self, text):
        raise AssertionError("the API was called")

    monkeypatch.setattr(OpenAIBackend, "embed_text", no_api)
    settings = config("--operator-encoder", "text", "--text-backend", "openai",
                      "--text-embedding-cache", str(tmp_path / "cache.json"))
    source = FixedEncoderSource("text", settings)
    cached = np.ones(256, dtype=np.float32) / 16
    source.encoder.backend.cache.set(f"{source.encoder.backend.id}|{to_pddl(BY_NAME['pickup'])}", cached)
    make_offline(source.encoder)
    points, unencodable = source.points("operator", REGISTERED[:2])
    assert [point.label for point in points] == ["pickup"] and np.array_equal(points[0].vector, cached)
    assert list(unencodable) == ["putdown"] and "--offline" in unencodable["putdown"]


@pytest.mark.unit
def test_an_expression_of_an_operator_and_literals():
    """pickup - eff (not (raised ?o)) + eff (raised ?o) is exactly pickup-raised in multi-hot (linear in its literals):
    the expression's one-literal operators keep pickup's name and parameters. A literal on a variable pickup does
    not have adds it as a parameter."""
    terms = parse_expression("pickup - eff (not (raised ?o)) + eff (raised ?o)", BY_NAME)
    assert [(sign, term) for sign, term, _ in terms] == [(1, "pickup"), (-1, "eff (not (raised ?o))"), (1, "eff (raised ?o)")]
    source = FixedEncoderSource("multi-hot", config("--operator-encoder", "multi-hot"))
    entries = [LabelledOperator(str(i), operator) for i, (_, _, operator) in enumerate(terms)]
    _, inputs, _ = source.encode(entries)
    total = sum(sign * row for (sign, _, _), row in zip(terms, inputs))
    np.testing.assert_array_equal(total, vectors(source, "operator")["pickup-raised"])
    (_, _, on), = parse_expression("pickup-raised + pre (on ?o ?u)", BY_NAME)[1:]
    assert [p.name for p in on.parameters] == ["?o", "?u"] and on.name == "pickup-raised"
    with pytest.raises(ValueError):
        parse_expression("pickup + (raised ?o)", BY_NAME)


@pytest.mark.unit
def test_a_custom_operator_keeps_the_layout_rules():
    """A predicate the environment cannot evaluate is unencodable in the compositional and multi-hot sources (with
    the encoder's reason), and still embedded as text."""
    painted = custom("painted", "(:action paint :parameters (?o - block) :precondition (and) :effect (and (painted ?o)))")
    for source in (compositional(), FixedEncoderSource("multi-hot", config("--operator-encoder", "multi-hot"))):
        assert "painted" in source.points("operator", [painted])[1]
    assert source.points("literal", [painted])[1]
    text = FixedEncoderSource("text", config("--operator-encoder", "text"))
    assert [p.label for p in text.points("operator", [painted])[0]] == ["painted"]


@pytest.mark.unit
def test_untrained_extractors_differ_by_seed_only():
    """The same seed gives the same untrained composition; another seed another."""
    first, again, other = (vectors(compositional("tree", seed), "operator") for seed in (0, 0, 1))
    assert all(np.array_equal(first[label], again[label]) for label in first)
    assert not np.allclose(first["pickup"], other["pickup"])


@pytest.mark.unit
def test_an_extractor_with_another_layout_than_its_config_is_refused():
    """A run saved with another layout than its config builds now (e.g. by older code) is refused, not misread."""
    source = compositional()
    with pytest.raises(ValueError):
        CompositionalSource("stale", source.config | {"max_operator_literals": 12}, source.extractor)
