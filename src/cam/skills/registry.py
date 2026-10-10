"""Maps skill names to Skill classes."""

from cam.skills.blocksworld_skills import (
    PickupRaisedSkill,
    PickupSkill,
    PlaceBesideSkill,
    PutdownSkill,
    StackSkill,
    UnstackRaisedSkill,
    UnstackSkill,
)
from cam.skills.skill import Skill

SKILL_REGISTRY: dict[str, type[Skill]] = {
    PickupSkill.name: PickupSkill,
    PutdownSkill.name: PutdownSkill,
    StackSkill.name: StackSkill,
    UnstackSkill.name: UnstackSkill,
    PickupRaisedSkill.name: PickupRaisedSkill,
    UnstackRaisedSkill.name: UnstackRaisedSkill,
    PlaceBesideSkill.name: PlaceBesideSkill,
}


def build_skill(skill_name: str, config: dict) -> Skill:
    if skill_name not in SKILL_REGISTRY:
        raise ValueError(f"unknown skill {skill_name!r}; known: {sorted(SKILL_REGISTRY)}")
    return SKILL_REGISTRY[skill_name](config)
