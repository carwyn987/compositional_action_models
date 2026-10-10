"""Place a held block on top of another block."""

from cam.skills.skill import Skill


class StackSkill(Skill):
    """stack(?o, ?u): ?o is the held block (picked up during setup), ?u the block it goes on. Setup picks
    up a random block and the episode's grounding picks a random clear block for ?u, so both blocks vary
    from episode to episode. Needs at least two blocks."""

    name = "stack"
    setup_skills = ("pickup",)
