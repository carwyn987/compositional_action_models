"""Load a skill's symbolic action model in a chosen representation format.

Each format has a directory next to this file holding one file per skill,
e.g. pddl/pickup.pddl.
"""

from importlib.resources import files
from importlib.resources.abc import Traversable

from cam.domain.operators.pddl_parser import parse_operator
from cam.domain.symbolic_action_model import SymbolicActionModel


def load_pddl_action_model(path: Traversable) -> SymbolicActionModel:
    return parse_operator(path.read_text())


# format -> (file extension, load(path) -> SymbolicActionModel)
SYMBOLIC_ACTION_MODEL_FORMATS = {
    "pddl": (".pddl", load_pddl_action_model),
}


def load_symbolic_action_model(skill_name: str, symbolic_action_model_format: str) -> SymbolicActionModel:
    if symbolic_action_model_format not in SYMBOLIC_ACTION_MODEL_FORMATS:
        raise ValueError(
            f"unknown symbolic action model format {symbolic_action_model_format!r}; "
            f"known: {sorted(SYMBOLIC_ACTION_MODEL_FORMATS)}"
        )
    extension, load = SYMBOLIC_ACTION_MODEL_FORMATS[symbolic_action_model_format]
    path = files(__package__) / symbolic_action_model_format / f"{skill_name}{extension}"
    if not path.is_file():
        raise FileNotFoundError(f"no {symbolic_action_model_format} action model for {skill_name!r}: {path}")
    return load(path)
