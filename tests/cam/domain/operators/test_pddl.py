import pytest

from cam.domain.operators.pddl import (
    Effect,
    PDDLOperator,
    Precondition,
    Predicate,
    TypedParameter,
)


def make_pickup() -> PDDLOperator:
    # (:action pickup :parameters (?o - block)
    #   :precondition (and (on-table ?o) (clear ?o) (gripper-empty) (not (holding ?o)))
    #   :effect (and (holding ?o) (not (on-table ?o)) (not (gripper-empty))))
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


@pytest.mark.unit
def test_create_pickup():
    op = make_pickup()
    assert op.name == "pickup"
    assert op.parameters[0].type == "block"
    assert len(op.preconditions) == 4
    assert [e.delete for e in op.effects] == [False, True, True]


@pytest.mark.unit
def test_negative_precondition():
    op = make_pickup()
    assert [p.negated for p in op.preconditions] == [False, False, False, True]
    assert op.preconditions[-1].predicate == Predicate("holding", ("?o",))


@pytest.mark.unit
def test_precondition_defaults_to_positive():
    assert Precondition(Predicate("clear", ("?o",))).negated is False
