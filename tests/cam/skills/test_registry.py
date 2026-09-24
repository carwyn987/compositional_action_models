import pytest

from cam.skills.registry import SKILL_REGISTRY, build_skill

PDDL_CONFIG = {"symbolic_action_model_format": "pddl"}


@pytest.mark.unit
@pytest.mark.parametrize("skill_name", sorted(SKILL_REGISTRY))
def test_every_registered_skill_builds(skill_name):
    skill = build_skill(skill_name, PDDL_CONFIG)
    assert skill.name == skill_name
    assert skill.symbolic_action_model.name == skill_name


@pytest.mark.unit
def test_unknown_skill_raises():
    with pytest.raises(ValueError):
        build_skill("not_a_skill", PDDL_CONFIG)
