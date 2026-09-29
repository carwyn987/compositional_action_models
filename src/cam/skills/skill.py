"""Abstract base class for skills."""

from abc import ABC, abstractmethod

from cam.domain.action_model_library.loader import load_symbolic_action_model
from cam.domain.operators.grounded_operator import GroundedOperator, applicable_groundings, ground
from cam.domain.operators.pddl import Predicate


class Skill(ABC):
    """A skill, described by a symbolic action model.

    The action model's representation is chosen by
    config["symbolic_action_model_format"] (e.g. "pddl"). The action model is
    lifted (e.g. pickup(?o)); ground() binds it to objects (e.g. pickup(block0)).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Skill name; also the name of its action model files."""

    def __init__(self, config: dict):
        self.config = config
        self.symbolic_action_model = load_symbolic_action_model(
            self.name, config["symbolic_action_model_format"]
        )

    def ground(self, binding: dict[str, str]) -> GroundedOperator:
        """This skill's action model with its parameters bound, e.g. {"?o": "block0"}."""
        return ground(self.symbolic_action_model, binding)

    def applicable_groundings(
        self, facts: frozenset[Predicate], objects: dict[str, str]
    ) -> list[GroundedOperator]:
        """Groundings of this skill's action model over objects (name -> type) whose
        preconditions hold in facts."""
        return applicable_groundings(self.symbolic_action_model, facts, objects)
