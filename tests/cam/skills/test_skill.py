import pytest

from cam.domain.operators.pddl import PDDLOperator
from cam.skills.pickup_skill import PickupSkill

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
