"""Tests for cam.domain.operators.pddl_parser.

Expected API:
    tokenize(text: str) -> list[str]
    parse_sexpr(text: str) -> list          # nested lists of str
    parse_operator(text: str) -> PDDLOperator
"""

import pytest

from cam.domain.operators import pddl_parser
from cam.domain.operators.pddl import (
    Effect,
    PDDLOperator,
    Precondition,
    Predicate,
    TypedParameter,
)
from tests.cam.domain.conftest import LEGACY_OPERATORS, PICKUP_PDDL

# --- tokenize ---------------------------------------------------------------


@pytest.mark.unit
def test_tokenize_splits_parens_and_symbols():
    assert pddl_parser.tokenize("(on ?o ?b)") == ["(", "on", "?o", "?b", ")"]


@pytest.mark.unit
def test_tokenize_ignores_whitespace_and_newlines():
    assert pddl_parser.tokenize("  (clear\n\t?o )  ") == ["(", "clear", "?o", ")"]


@pytest.mark.unit
def test_tokenize_keeps_keywords_and_type_dash():
    assert pddl_parser.tokenize("(:parameters (?o - block))") == [
        "(", ":parameters", "(", "?o", "-", "block", ")", ")",
    ]


# --- parse_sexpr ------------------------------------------------------------


@pytest.mark.unit
def test_parse_sexpr_nested():
    assert pddl_parser.parse_sexpr("(a (b ?x) (c))") == ["a", ["b", "?x"], ["c"]]


@pytest.mark.unit
@pytest.mark.parametrize("text", ["(a (b)", "(a))", ")", ""])
def test_parse_sexpr_rejects_unbalanced_or_empty(text):
    with pytest.raises(ValueError):
        pddl_parser.parse_sexpr(text)


@pytest.mark.unit
def test_parse_sexpr_rejects_multiple_top_level_expressions():
    with pytest.raises(ValueError):
        pddl_parser.parse_sexpr("(a) (b)")


# --- parse_operator ---------------------------------------------------------


@pytest.mark.unit
def test_parse_pickup_matches_hand_built(pickup_operator):
    assert pddl_parser.parse_operator(PICKUP_PDDL) == pickup_operator


@pytest.mark.unit
def test_parse_multiple_typed_parameters():
    op = pddl_parser.parse_operator(LEGACY_OPERATORS["stack"])
    assert op.parameters == (TypedParameter("?o", "block"), TypedParameter("?b", "block"))


@pytest.mark.unit
def test_parse_single_literal_precondition_without_and():
    op = pddl_parser.parse_operator(LEGACY_OPERATORS["putdown"])
    assert op.preconditions == (Precondition(Predicate("holding", ("?o",))),)


@pytest.mark.unit
def test_parse_single_literal_effect_without_and():
    op = pddl_parser.parse_operator(LEGACY_OPERATORS["pushleft"])
    assert op.effects == (Effect(Predicate("moved-left", ("?o",))),)


@pytest.mark.unit
def test_parse_zero_arity_predicate():
    op = pddl_parser.parse_operator(LEGACY_OPERATORS["pickup"])
    assert Precondition(Predicate("gripper-empty", ())) in op.preconditions


@pytest.mark.unit
def test_parse_full_unstack():
    assert pddl_parser.parse_operator(LEGACY_OPERATORS["unstack"]) == PDDLOperator(
        name="unstack",
        parameters=(TypedParameter("?o", "block"), TypedParameter("?b", "block")),
        preconditions=(
            Precondition(Predicate("on", ("?o", "?b"))),
            Precondition(Predicate("clear", ("?o",))),
            Precondition(Predicate("gripper-empty", ())),
        ),
        effects=(
            Effect(Predicate("on-table", ("?o",))),
            Effect(Predicate("clear", ("?b",))),
            Effect(Predicate("on", ("?o", "?b")), delete=True),
        ),
    )


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(LEGACY_OPERATORS))
def test_parse_all_legacy_operators(name):
    op = pddl_parser.parse_operator(LEGACY_OPERATORS[name])
    assert op.name == name


# --- parse_operator errors --------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "(:predicate foo)",  # not an :action
        "(:action a :parameters (?o - block) :effect (p ?o))",  # missing :precondition
        "(:action a :parameters (?o - block) :precondition (p ?o))",  # missing :effect
        "(:action a :parameters (?o - block) :precondition (p ?o) :effect (q ?o) :bogus (r))",
        "(:action a :parameters (?o block) :precondition (p ?o) :effect (q ?o))",  # missing '-'
    ],
)
def test_parse_operator_rejects_malformed(text):
    with pytest.raises(ValueError):
        pddl_parser.parse_operator(text)
