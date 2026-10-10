import pytest

from cam.domain.pddl.pddl import PDDLOperator
from cam.domain.symbols import Predicate
from cam.skills.blocksworld_skills import PickupSkill, PutdownSkill

PDDL_CONFIG = {"symbolic_action_model_format": "pddl"}


@pytest.mark.unit
def test_pickup_loads_its_pddl_action_model():
    skill = PickupSkill(PDDL_CONFIG)
    assert skill.name == "pickup"
    assert skill.config is PDDL_CONFIG
    assert isinstance(skill.symbolic_action_model, PDDLOperator)
    assert skill.symbolic_action_model.name == "pickup"


@pytest.mark.unit
def test_unknown_symbolic_action_model_format_raises():
    with pytest.raises(ValueError):
        PickupSkill({"symbolic_action_model_format": "not_a_format"})


@pytest.mark.unit
def test_ground_binds_the_skill_action_model():
    """skill.ground(binding) grounds the skill's own action model."""
    skill = PickupSkill(PDDL_CONFIG)
    grounded = skill.ground({"?o": "block0"})
    assert grounded.operator is skill.symbolic_action_model
    assert str(grounded) == "pickup(block0)"


@pytest.mark.unit
def test_applicable_groundings_use_the_skill_action_model():
    """skill.applicable_groundings returns groundings of the skill's own action model."""
    facts = frozenset({Predicate("holding", ("block1",))})
    groundings = PutdownSkill(PDDL_CONFIG).applicable_groundings(facts, {"block0": "block", "block1": "block"})
    assert [str(g) for g in groundings] == ["putdown(block1)"]
