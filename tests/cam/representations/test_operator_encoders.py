import dataclasses

import numpy as np
import pytest

from cam.domain.pddl.pddl import Precondition
from cam.domain.pddl.pddl_parser import parse_operator
from cam.domain.symbols import Predicate
from cam.representations.one_hot_operator_encoder import OneHotOperatorEncoder
from cam.representations.pddl_multi_hot_operator_encoder import PDDLMultiHotOperatorEncoder
from cam.representations.random_operator_encoder import RandomOperatorEncoder
from tests.cam.domain.conftest import LEGACY_OPERATORS

PREDICATE_ARITIES = {"holding": 1, "on-table": 1, "clear": 1, "on": 2, "gripper-empty": 0}
PICKUP = parse_operator(LEGACY_OPERATORS["pickup"])
PUTDOWN = parse_operator(LEGACY_OPERATORS["putdown"])
STACK = parse_operator(LEGACY_OPERATORS["stack"])


def encoder(max_arity: int = 2) -> PDDLMultiHotOperatorEncoder:
    return PDDLMultiHotOperatorEncoder(PREDICATE_ARITIES, max_arity)


def operator(effect: str):
    return parse_operator(f"(:action op :parameters (?x - block ?y - block) :precondition (and) :effect (and {effect}))")


@pytest.mark.unit
@pytest.mark.parametrize("max_arity, dim", [(1, 36), (2, 56), (3, 84)])
def test_vocabulary_size_depends_only_on_predicates_and_max_arity(max_arity, dim):
    """4 sections x (1 name-only + positional features over distinct parameter positions) per predicate."""
    assert encoder(max_arity).dim == dim


@pytest.mark.unit
def test_different_operators_get_different_encodings():
    """pickup and putdown share no literals, so their multi-hot vectors differ."""
    assert not np.array_equal(encoder().encode(PICKUP), encoder().encode(PUTDOWN))


@pytest.mark.unit
def test_variable_names_do_not_matter():
    """Renaming parameters leaves the encoding unchanged: features use positions, not names."""
    renamed = parse_operator(LEGACY_OPERATORS["stack"].replace("?o", "?first").replace("?b", "?second"))
    np.testing.assert_array_equal(encoder().encode(STACK), encoder().encode(renamed))


@pytest.mark.unit
def test_argument_order_is_encoded():
    """(on ?x ?y) and (on ?y ?x) differ in their positional feature but share the name-only feature."""
    a, b = encoder().encode(operator("(on ?x ?y)")), encoder().encode(operator("(on ?y ?x)"))
    assert not np.array_equal(a, b)
    assert a[encoder().index[("add", "on")]] == b[encoder().index[("add", "on")]] == 1


@pytest.mark.unit
def test_repaired_operator_changes_only_its_new_features():
    """Adding a precondition sets exactly that literal's two features; the rest of the encoding is unchanged."""
    repaired = dataclasses.replace(
        PICKUP, preconditions=PICKUP.preconditions + (Precondition(Predicate("holding", ("?o",)), negated=True),)
    )
    before, after = encoder().encode(PICKUP), encoder().encode(repaired)
    changed = {encoder().features[i] for i in np.flatnonzero(before != after)}
    assert changed == {("negated_precondition", "holding", (0,)), ("negated_precondition", "holding")}


@pytest.mark.unit
@pytest.mark.parametrize(
    "operator_text",
    [
        "(:action op :parameters (?x - block) :precondition (and) :effect (and (painted ?x)))",  # unknown predicate
        "(:action op :parameters (?x - block) :precondition (and) :effect (and (on ?x ?x)))",  # repeated parameter
        "(:action op :parameters (?x - block ?y - block ?z - block) :precondition (and) :effect (and))",  # arity 3 > 2
    ],
    ids=["unknown-predicate", "repeated-parameter", "too-many-parameters"],
)
def test_operators_outside_the_vocabulary_are_rejected(operator_text):
    """Encoding refuses operators it cannot represent rather than silently dropping features."""
    with pytest.raises(ValueError):
        encoder().encode(parse_operator(operator_text))


@pytest.mark.unit
def test_one_hot_baseline_marks_training_operators_only():
    """The baseline gives each training operator its own entry and rejects any other operator."""
    one_hot = OneHotOperatorEncoder([PICKUP, PUTDOWN])
    np.testing.assert_array_equal(one_hot.encode(PUTDOWN), [0, 1])
    with pytest.raises(ValueError):
        one_hot.encode(STACK)


@pytest.mark.unit
def test_random_vectors_are_fixed_per_operator_and_seed():
    """The same operator and seed always give the same unit vector (also across encoder instances);
    other operators or seeds give different vectors."""
    vector = RandomOperatorEncoder(128, seed=0).encode(PICKUP)
    assert vector.shape == (128,) and np.linalg.norm(vector) == pytest.approx(1.0)
    np.testing.assert_array_equal(vector, RandomOperatorEncoder(128, seed=0).encode(PICKUP))
    assert not np.allclose(vector, RandomOperatorEncoder(128, seed=0).encode(PUTDOWN))
    assert not np.allclose(vector, RandomOperatorEncoder(128, seed=1).encode(PICKUP))


@pytest.mark.unit
def test_random_vectors_ignore_operator_structure():
    """Operators are identified by name: a changed operator that keeps its name keeps its vector,
    and any operator (seen or not) can be encoded."""
    repaired = dataclasses.replace(PICKUP, effects=PICKUP.effects[:1])
    encoder = RandomOperatorEncoder(64)
    np.testing.assert_array_equal(encoder.encode(PICKUP), encoder.encode(repaired))
    assert encoder.encode(STACK).shape == (64,)
