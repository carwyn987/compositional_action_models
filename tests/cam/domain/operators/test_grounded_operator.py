import pytest

from cam.domain.operators.grounded_operator import GroundedOperator, ground
from cam.domain.operators.pddl import Predicate
from cam.domain.operators.pddl_parser import parse_operator
from cam.domain.symbolic_action_model import applicable_groundings, groundings
from tests.cam.domain.conftest import LEGACY_OPERATORS

# pickup_operator (conftest): pre (on-table ?o) (clear ?o) (gripper-empty) (not (holding ?o));
# effects (holding ?o) (not (on-table ?o)) (not (gripper-empty))
BLOCK0 = {"?o": "block0"}
BLOCK0_ON_TABLE = frozenset(
    {Predicate("on-table", ("block0",)), Predicate("clear", ("block0",)), Predicate("gripper-empty", ())}
)
BLOCK0_HELD = frozenset({Predicate("holding", ("block0",))})


@pytest.mark.unit
def test_ground_substitutes_variables(pickup_operator):
    """Grounding replaces ?o with the bound object in every literal."""
    grounded = ground(pickup_operator, BLOCK0)
    assert grounded.arguments == ("block0",)
    assert grounded.preconditions[0].predicate == Predicate("on-table", ("block0",))
    assert grounded.effects[0].predicate == Predicate("holding", ("block0",))
    assert str(grounded) == "pickup(block0)"


@pytest.mark.unit
@pytest.mark.parametrize("binding", [{}, {"?o": "block0", "?b": "block1"}])
def test_ground_requires_exact_binding(pickup_operator, binding):
    """A binding must bind every parameter and nothing else."""
    with pytest.raises(ValueError):
        ground(pickup_operator, binding)


@pytest.mark.unit
def test_preconditions_hold_when_all_facts_present(pickup_operator):
    """pickup(block0) is applicable to a block resting on the table with an empty gripper."""
    assert ground(pickup_operator, BLOCK0).preconditions_hold(BLOCK0_ON_TABLE)


@pytest.mark.unit
def test_preconditions_fail_when_a_fact_is_missing(pickup_operator):
    """pickup(block0) is not applicable while the gripper holds the block."""
    assert not ground(pickup_operator, BLOCK0).preconditions_hold(BLOCK0_HELD)


@pytest.mark.unit
def test_negated_precondition_fails_when_fact_present(pickup_operator):
    """(not (holding ?o)) blocks pickup(block0) if block0 is also reported held."""
    facts = BLOCK0_ON_TABLE | {Predicate("holding", ("block0",))}
    assert not ground(pickup_operator, BLOCK0).preconditions_hold(facts)


@pytest.mark.unit
def test_binding_selects_the_object(pickup_operator):
    """Facts about block0 do not make pickup(block1) applicable."""
    assert not ground(pickup_operator, {"?o": "block1"}).preconditions_hold(BLOCK0_ON_TABLE)


@pytest.mark.unit
def test_effects_hold_after_pickup(pickup_operator):
    """Holding block0 satisfies pickup(block0)'s effects."""
    assert ground(pickup_operator, BLOCK0).effects_hold(BLOCK0_HELD)


@pytest.mark.unit
def test_effects_fail_while_a_deleted_fact_remains(pickup_operator):
    """pickup deletes (on-table ?o): holding a block still reported on-table is not success."""
    facts = BLOCK0_HELD | {Predicate("on-table", ("block0",))}
    assert not ground(pickup_operator, BLOCK0).effects_hold(facts)


@pytest.mark.unit
def test_groundings_match_parameter_types(pickup_operator):
    """Only objects whose type matches the parameter type are bound."""
    objects = {"block0": "block", "block1": "block", "tray0": "tray"}
    assert [str(g) for g in groundings(pickup_operator, objects)] == ["pickup(block0)", "pickup(block1)"]


@pytest.mark.unit
def test_groundings_bind_distinct_objects():
    """A two-parameter operator is never bound to the same object twice."""
    stack = parse_operator(LEGACY_OPERATORS["stack"])
    objects = {"block0": "block", "block1": "block"}
    assert [str(g) for g in groundings(stack, objects)] == ["stack(block0, block1)", "stack(block1, block0)"]


@pytest.mark.unit
def test_applicable_groundings_filter_by_preconditions(pickup_operator):
    """With block0 resting on the table and block1 held, only pickup(block0) is applicable."""
    objects = {"block0": "block", "block1": "block"}
    facts = BLOCK0_ON_TABLE | {Predicate("holding", ("block1",))}
    assert [str(g) for g in applicable_groundings(pickup_operator, facts, objects)] == ["pickup(block0)"]


@pytest.mark.unit
def test_direct_construction_requires_one_argument_per_parameter(pickup_operator):
    """GroundedOperator itself rejects an argument count that does not match the operator."""
    with pytest.raises(ValueError):
        GroundedOperator(pickup_operator, ("block0", "block1"))


@pytest.mark.unit
def test_groundings_are_equal_and_hashable_by_operator_and_arguments(pickup_operator):
    """Two groundings of the same operator on the same objects are interchangeable (e.g. as set members)."""
    assert ground(pickup_operator, BLOCK0) == GroundedOperator(pickup_operator, ("block0",))
    assert len({ground(pickup_operator, BLOCK0), ground(pickup_operator, BLOCK0)}) == 1
