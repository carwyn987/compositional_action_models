"""Lift a block off the block it is stacked on."""

from cam.skills.skill import Skill


class UnstackSkill(Skill):
    """unstack(?o, ?u): lift ?o off ?u. Setup picks up a random block and stacks it on a random other
    block, so the stacked pair varies from episode to episode. Needs at least two blocks."""

    name = "unstack"
    setup_skills = ("pickup", "stack")
