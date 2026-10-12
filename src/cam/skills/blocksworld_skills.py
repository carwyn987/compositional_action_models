"""The blocksworld skills. Each is its name (which also names its action model files) and its setup
chain; everything else is in Skill. Fetch resets with every block on the table, so skills that start
from a held or stacked block reach that state through setup_skills."""

from cam.skills.skill import Skill


class PickupSkill(Skill):
    """Pick a block up off the table."""

    name = "pickup"


class PutdownSkill(Skill):
    """Put a held block down on the table. Each episode starts from either setup at random: pickup (the block
    held just off the table) or pickup-raised (held 30 cm up, so it must be lowered before letting go: a drop
    is a failure)."""

    name = "putdown"
    setup_skills = ("pickup",)
    setup_alternatives = (("pickup-raised",),)


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


# Variants of the skills above, each one literal (or one parameter and literal) away from its original,
# for repair and new-skill experiments.


class PickupRaisedSkill(Skill):
    """pickup + effect (raised ?o): pick a block up and hold it high."""

    name = "pickup-raised"


class UnstackRaisedSkill(Skill):
    """unstack + effect (raised ?o): lift a block off another and hold it high (a repair of unstack)."""

    name = "unstack-raised"
    setup_skills = ("pickup", "stack")


class PlaceBesideSkill(Skill):
    """putdown + parameter ?u + effect (beside ?o ?u): put the held block on the table next to block ?u.
    Setup picks up a random block and the grounding picks a random block on the table for ?u."""

    name = "place-beside"
    setup_skills = ("pickup",)



# Pushes along the table into global regions (at-left / at-right / at-front / at-back: 10 cm from the centre of
# the reset area) or next to another block, starting with the gripper empty and the block on the table.


class PushLeftSkill(Skill):
    name = "push-left"


class PushRightSkill(Skill):
    name = "push-right"


class PushFrontSkill(Skill):
    name = "push-front"


class PushBackSkill(Skill):
    name = "push-back"


class PushLeftFrontSkill(Skill):
    """push-left + effect (at-front ?o): the diagonal, combining two pushes' goals (held out for testing)."""

    name = "push-left-front"


class PushBesideSkill(Skill):
    """Push block ?o along the table until it is beside block ?u (teaches beside without carrying)."""

    name = "push-beside"
