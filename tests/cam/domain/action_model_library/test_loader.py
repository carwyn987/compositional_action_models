import pytest

from cam.domain.action_model_library.loader import load_symbolic_action_model
from cam.domain.operators.pddl import PDDLOperator


@pytest.mark.unit
def test_load_pickup_pddl():
    op = load_symbolic_action_model("pickup", "pddl")
    assert isinstance(op, PDDLOperator)
    assert op.name == "pickup"


@pytest.mark.unit
def test_unknown_format_raises():
    with pytest.raises(ValueError):
        load_symbolic_action_model("pickup", "not_a_format")


@pytest.mark.unit
def test_missing_skill_file_raises():
    with pytest.raises(FileNotFoundError):
        load_symbolic_action_model("not_a_skill", "pddl")
