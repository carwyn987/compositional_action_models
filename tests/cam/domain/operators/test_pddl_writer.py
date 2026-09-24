"""Tests for cam.domain.operators.pddl_writer.

Canonical form produced by to_pddl:
    - one line, single spaces
    - :precondition and :effect always wrapped in (and ...), even for 0 or 1 literal
    - every parameter carries its own type: (?o - block ?b - block)

Together with the parser this closes the loop:
    PDDLOperator -> string -> PDDLOperator   (op round trip)
    string -> PDDLOperator -> string         (canonical strings are fixed points)
"""

import pytest

from cam.domain.operators.pddl import (
    Effect,
    PDDLOperator,
    Precondition,
    Predicate,
    TypedParameter,
)
from cam.domain.operators.pddl_parser import parse_operator
from cam.domain.operators.pddl_writer import to_pddl
from tests.cam.domain.conftest import LEGACY_OPERATORS, PICKUP_PDDL

# Canonical renderings of two legacy operators whose source is not canonical.
CANONICAL_PUTDOWN = (
    "(:action putdown :parameters (?o - block) "
    ":precondition (and (holding ?o)) "
    ":effect (and (on-table ?o) (clear ?o) (gripper-empty) (not (holding ?o))))"
)
CANONICAL_PUSHLEFT = (
    "(:action pushleft :parameters (?o - block) "
    ":precondition (and (on-table ?o) (gripper-empty)) "
    ":effect (and (moved-left ?o)))"
)


# --- exact output -----------------------------------------------------------


@pytest.mark.unit
def test_write_pickup_exact(pickup_operator):
    assert to_pddl(pickup_operator) == PICKUP_PDDL


@pytest.mark.unit
def test_single_literal_is_wrapped_in_and():
    assert to_pddl(parse_operator(LEGACY_OPERATORS["putdown"])) == CANONICAL_PUTDOWN
    assert to_pddl(parse_operator(LEGACY_OPERATORS["pushleft"])) == CANONICAL_PUSHLEFT


@pytest.mark.unit
def test_empty_conditions_and_parameters():
    op = PDDLOperator(name="noop", parameters=(), preconditions=(), effects=())
    assert to_pddl(op) == "(:action noop :parameters () :precondition (and) :effect (and))"


@pytest.mark.unit
def test_zero_arity_predicate():
    op = PDDLOperator(
        name="release",
        parameters=(),
        preconditions=(Precondition(Predicate("gripper-empty", ()), negated=True),),
        effects=(Effect(Predicate("gripper-empty", ())),),
    )
    assert to_pddl(op) == (
        "(:action release :parameters () "
        ":precondition (and (not (gripper-empty))) "
        ":effect (and (gripper-empty)))"
    )


@pytest.mark.unit
def test_every_parameter_carries_its_type():
    op = PDDLOperator(
        name="stack",
        parameters=(TypedParameter("?o", "block"), TypedParameter("?b", "block")),
        preconditions=(),
        effects=(),
    )
    assert ":parameters (?o - block ?b - block)" in to_pddl(op)


@pytest.mark.unit
def test_grouped_parameters_are_expanded():
    grouped = "(:action stack :parameters (?o ?b - block) :precondition (and) :effect (and))"
    assert to_pddl(parse_operator(grouped)) == (
        "(:action stack :parameters (?o - block ?b - block) :precondition (and) :effect (and))"
    )


# --- PDDLOperator -> string -> PDDLOperator ---------------------------------


@pytest.mark.unit
def test_operator_round_trip_pickup(pickup_operator):
    assert parse_operator(to_pddl(pickup_operator)) == pickup_operator


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(LEGACY_OPERATORS))
def test_operator_round_trip_legacy(name):
    op = parse_operator(LEGACY_OPERATORS[name])
    assert parse_operator(to_pddl(op)) == op


# --- string -> PDDLOperator -> string ---------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [PICKUP_PDDL, CANONICAL_PUTDOWN, CANONICAL_PUSHLEFT],
    ids=["pickup", "putdown", "pushleft"],
)
def test_canonical_string_round_trip(text):
    assert to_pddl(parse_operator(text)) == text


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(LEGACY_OPERATORS))
def test_canonicalization_is_idempotent(name):
    once = to_pddl(parse_operator(LEGACY_OPERATORS[name]))
    assert to_pddl(parse_operator(once)) == once


@pytest.mark.unit
def test_formatting_does_not_change_canonical_form():
    messy = """
        (:action   pickup
            :parameters ( ?o  -  block )
            :precondition (and (on-table ?o) (clear ?o)
                               (gripper-empty) (not (holding ?o)))
            :effect (and (holding ?o)
                         (not (on-table ?o)) (not (gripper-empty)))
        )
    """
    assert to_pddl(parse_operator(messy)) == PICKUP_PDDL
