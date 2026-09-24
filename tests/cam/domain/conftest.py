import pytest

from cam.domain.operators.pddl import (
    Effect,
    PDDLOperator,
    Precondition,
    Predicate,
    TypedParameter,
)

PICKUP_PDDL = (
    "(:action pickup :parameters (?o - block) "
    ":precondition (and (on-table ?o) (clear ?o) (gripper-empty) (not (holding ?o))) "
    ":effect (and (holding ?o) (not (on-table ?o)) (not (gripper-empty))))"
)


# Operator strings from deprecated/fetch_blockworld/skills.py.
LEGACY_OPERATORS = {
    "pickup": (
        "(:action pickup :parameters (?o - block) "
        ":precondition (and (on-table ?o) (clear ?o) (gripper-empty)) "
        ":effect (and (holding ?o) (not (on-table ?o)) (not (gripper-empty))))"
    ),
    "putdown": (
        "(:action putdown :parameters (?o - block) "
        ":precondition (holding ?o) "
        ":effect (and (on-table ?o) (clear ?o) (gripper-empty) (not (holding ?o))))"
    ),
    "pushleft": (
        "(:action pushleft :parameters (?o - block) "
        ":precondition (and (on-table ?o) (gripper-empty)) "
        ":effect (moved-left ?o))"
    ),
    "pushright": (
        "(:action pushright :parameters (?o - block) "
        ":precondition (and (on-table ?o) (gripper-empty)) "
        ":effect (moved-right ?o))"
    ),
    "pushforward": (
        "(:action pushforward :parameters (?o - block) "
        ":precondition (and (on-table ?o) (gripper-empty)) "
        ":effect (moved-forward ?o))"
    ),
    "pushbackward": (
        "(:action pushbackward :parameters (?o - block) "
        ":precondition (and (on-table ?o) (gripper-empty)) "
        ":effect (moved-backward ?o))"
    ),
    "reach_top": (
        "(:action reach_top :parameters (?o - block) "
        ":precondition (clear ?o) "
        ":effect (gripper-at-top ?o))"
    ),
    "stack": (
        "(:action stack :parameters (?o - block ?b - block) "
        ":precondition (and (holding ?o) (clear ?b)) "
        ":effect (and (on ?o ?b) (clear ?o) (gripper-empty) "
        "(not (holding ?o)) (not (clear ?b))))"
    ),
    "unstack": (
        "(:action unstack :parameters (?o - block ?b - block) "
        ":precondition (and (on ?o ?b) (clear ?o) (gripper-empty)) "
        ":effect (and (on-table ?o) (clear ?b) (not (on ?o ?b))))"
    ),
}


@pytest.fixture
def pickup_operator() -> PDDLOperator:
    """The PDDLOperator equivalent of PICKUP_PDDL."""
    return PDDLOperator(
        name="pickup",
        parameters=(TypedParameter("?o", "block"),),
        preconditions=(
            Precondition(Predicate("on-table", ("?o",))),
            Precondition(Predicate("clear", ("?o",))),
            Precondition(Predicate("gripper-empty", ())),
            Precondition(Predicate("holding", ("?o",)), negated=True),
        ),
        effects=(
            Effect(Predicate("holding", ("?o",))),
            Effect(Predicate("on-table", ("?o",)), delete=True),
            Effect(Predicate("gripper-empty", ()), delete=True),
        ),
    )
