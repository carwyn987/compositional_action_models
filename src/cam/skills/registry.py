"""Maps skill names to Skill classes."""

from cam.skills.blocksworld_skills import (
    PickupRaisedSkill,
    PickupSkill,
    PlaceBesideSkill,
    PushBackSkill,
    PushBesideSkill,
    PushFrontSkill,
    PushLeftBackSkill,
    PushLeftFrontSkill,
    PushLeftSkill,
    PushRightBackSkill,
    PushRightFrontSkill,
    PushRightSkill,
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
    PushLeftSkill.name: PushLeftSkill,
    PushRightSkill.name: PushRightSkill,
    PushFrontSkill.name: PushFrontSkill,
    PushBackSkill.name: PushBackSkill,
    PushLeftFrontSkill.name: PushLeftFrontSkill,
    PushRightFrontSkill.name: PushRightFrontSkill,
    PushLeftBackSkill.name: PushLeftBackSkill,
    PushRightBackSkill.name: PushRightBackSkill,
    PushBesideSkill.name: PushBesideSkill,
}


def build_skill(skill_name: str, config: dict) -> Skill:
    if skill_name not in SKILL_REGISTRY:
        raise ValueError(f"unknown skill {skill_name!r}; known: {sorted(SKILL_REGISTRY)}")
    return SKILL_REGISTRY[skill_name](config)
