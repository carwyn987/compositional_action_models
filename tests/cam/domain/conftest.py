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
