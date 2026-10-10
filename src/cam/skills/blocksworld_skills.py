"""The blocksworld skills. Each is its name (which also names its action model files) and its setup
chain; everything else is in Skill. Fetch resets with every block on the table, so skills that start
from a held or stacked block reach that state through setup_skills."""

from cam.skills.skill import Skill


class PickupSkill(Skill):
    """Pick a block up off the table."""

    name = "pickup"


class PutdownSkill(Skill):
    """Put a held block down on the table."""

    name = "putdown"
    setup_skills = ("pickup",)


class StackSkill(Skill):
    """stack(?o, ?u): place the held block ?o on block ?u. Setup picks up a random block and the episode's
    grounding picks a random clear block for ?u, so both vary between episodes. Needs at least two blocks."""

    name = "stack"
    setup_skills = ("pickup",)


class UnstackSkill(Skill):
    """unstack(?o, ?u): lift block ?o off block ?u. Setup picks up a random block and stacks it on a random
    other block, so the stacked pair varies between episodes. Needs at least two blocks."""

    name = "unstack"
    setup_skills = ("pickup", "stack")
