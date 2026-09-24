import pytest

from cam.domain.operators.pddl import Precondition, Predicate


@pytest.mark.unit
def test_create_pickup(pickup_operator):
    op = pickup_operator
    assert op.name == "pickup"
    assert op.parameters[0].type == "block"
    assert len(op.preconditions) == 4
    assert [e.delete for e in op.effects] == [False, True, True]


@pytest.mark.unit
def test_negative_precondition(pickup_operator):
    op = pickup_operator
    assert [p.negated for p in op.preconditions] == [False, False, False, True]
    assert op.preconditions[-1].predicate == Predicate("holding", ("?o",))


@pytest.mark.unit
def test_precondition_defaults_to_positive():
    assert Precondition(Predicate("clear", ("?o",))).negated is False
