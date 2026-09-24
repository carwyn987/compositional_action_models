"""Maps skill names to Skill classes."""

from cam.skills.pickup_skill import PickupSkill
from cam.skills.skill import Skill

SKILL_REGISTRY: dict[str, type[Skill]] = {
    PickupSkill.name: PickupSkill,
}


def build_skill(skill_name: str, config: dict) -> Skill:
    if skill_name not in SKILL_REGISTRY:
        raise ValueError(f"unknown skill {skill_name!r}; known: {sorted(SKILL_REGISTRY)}")
    return SKILL_REGISTRY[skill_name](config)
