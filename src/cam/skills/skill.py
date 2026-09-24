"""Abstract base class for skills."""

from abc import ABC, abstractmethod

from cam.domain.action_model_library.loader import load_symbolic_action_model


class Skill(ABC):
    """A skill, described by a symbolic action model.

    The action model's representation is chosen by
    config["symbolic_action_model_format"] (e.g. "pddl").
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
